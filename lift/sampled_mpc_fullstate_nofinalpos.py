# This script is used to sample the Conditional ODE model for the Two Arm Lift task and execute the demo.
# It uses the 3-dimensional rotation vector of the arm's state and action.

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl
import copy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from utils.conditional_Action_DiT import Conditional_ODE
from utils.env import TwoArmLiftRole
from scipy.spatial.transform import Rotation as R
from utils.transform_utils import SE3_log_map, SE3_exp_map, quat_to_rot6d, rotvec_to_rot6d, rot6d_to_quat, rot6d_to_rotvec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    result = []

    for traj in expert_data:
        for start_idx in range(n_subtraj):
            # If not enough steps, pad with the last step
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                # Need padding
                sub_traj = traj[start_idx:]
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)

    result = np.stack(result, axis=0)
    return result

class PolicyPlayer:
    def __init__ (self, env, render = False):
        self.env = env

        self.control_freq = env.control_freq
        self.dt = 1.0 / self.control_freq
        self.max_time = 10
        self.max_steps = int(self.max_time / self.dt)

        self.render = render

        # Extract the base position and orientation (quaternion) from the simulation data
        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))

        robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
        self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
        self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3,3))

        # Rotation matrix of robots for the home position, both in their own base frame
        self.R_be_home = np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, -1]])

        self.n_action = self.env.action_spec[0].shape[0]

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset

    def reset(self, seed = 0):
        """
        Resetting environment. Re-initializing the waypoints too.
        """
        np.random.seed(seed)
        obs = self.env.reset()

        # Setting up constants
        self.pot_handle_offset_z = 0.012
        self.pot_handle_offset_x = 0.015
        self.pot_handle_offset = np.array([self.pot_handle_offset_x, 0, self.pot_handle_offset_z])
        self.pot_handle0_pos = self.robot0_base_ori_rotm.T @ (self.env._handle0_xpos - self.robot0_base_pos) + self.pot_handle_offset
        self.pot_handle1_pos = self.robot1_base_ori_rotm.T @ (self.env._handle1_xpos - self.robot1_base_pos) + self.pot_handle_offset
        jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
        jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint") #gripper0_right_finger_joint, gripper0_right_right_outer_knuckle_joint
        self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]

        self.rollout = {}
        self.rollout["observations"] = []
        self.rollout["actions"] = []

        return obs
        
    def load_model(self, type = "rotvec", state_dim = 7, action_dim = 7):
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 25 # horizon, length of each trajectory
        T = 250 # total time steps

        # Load expert data
        expert_data = np.load("data/expert_actions_rotvec_20.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]
        orig1 = expert_data1.copy()
        orig2 = expert_data2.copy()
        expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
        expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

        # Compute mean and standard deviation
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        # Normalize data
        expert_data1 = (expert_data1 - mean) / std
        expert_data2 = (expert_data2 - mean) / std
        orig1 = (orig1 - mean) / std
        orig2 = (orig2 - mean) / std

        env = TwoArmLift(state_size=state_dim, action_size=action_dim)

        # Prepare conditional vectors for training
        with open("data/pot_states_rotvec_20.npy", "rb") as f:
            obs = np.load(f)
        obs_init1 = expert_data1[:, 0, :]
        obs_init2 = expert_data2[:, 0, :]
        obs_final1 = np.repeat(orig1[:, -1, :], repeats=T, axis=0)
        obs_final2 = np.repeat(orig2[:, -1, :], repeats=T, axis=0)
        # obs_init1_cond = expert_data1[:, 1, :3]
        obs = np.repeat(obs, repeats=T, axis=0)
        obs1 = np.hstack([obs_init1, obs_init2, obs])
        obs2 = np.hstack([obs_init2, obs_init1, obs])
        obs1 = torch.FloatTensor(obs1).to(device)
        obs2 = torch.FloatTensor(obs2).to(device)
        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]

        # Preparing expert data for training
        actions1 = expert_data1[:, :H, :]
        actions2 = expert_data2[:, :H, :]
        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        # Load the model
        action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra="_lift_mpc_P25E1_crosscond_nofinalpos_fullstate_nolf")

        return action_cond_ode
    
    
    def reactive_mpc_plan(self, ode_model, initial_states, final_states, obs, segment_length=25, total_steps=250, n_implement=5):
        """
        Plans a full trajectory (total_steps long) by iteratively planning
        segment_length-steps using the diffusion model and replanning at every timestep.
        
        Parameters:
        - ode_model: the Conditional_ODE (diffusion model) instance.
        - env: your environment, which must implement reset_to() and step().
        - initial_state: a numpy array of shape (state_size,) (the current state).
        - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
        - model_i: the index of the agent/model being planned for.
        - segment_length: number of timesteps to plan in each segment.
        - total_steps: total length of the planned trajectory.
        
        Returns:
        - full_traj: a numpy array of shape (total_steps, state_size)
        """
        full_traj = []
        current_states = initial_states.copy()

        for seg in range(total_steps // n_implement):
            segments = []
            base_states = current_states.copy()
            for i in range(len(current_states)):
                cond = [base_states[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(base_states[j])
                cond.append(obs)
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            seg_array = np.stack(segments, axis=0)
            full_traj.append(seg_array)

        full_traj = np.concatenate(full_traj, axis=1) 
        print("Full trajectory shape: ", np.shape(full_traj))
        return np.array(full_traj)

    
    def get_demo(self, seed, cond_idx, H=25, T=250):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed)

        # Loading
        expert_data = np.load("data/expert_actions_rotvec_20.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]
        orig1 = expert_data1.copy()
        orig2 = expert_data2.copy()
        expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
        expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)
        combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
        mean = np.mean(combined_data, axis=(0,1))
        std = np.std(combined_data, axis=(0,1))

        # Normalize data
        expert_data1 = (expert_data1 - mean) / std
        expert_data2 = (expert_data2 - mean) / std
        orig1 = (orig1 - mean) / std
        orig2 = (orig2 - mean) / std

        obs_init1 = expert_data1[:, 0, :]
        obs_init2 = expert_data2[:, 0, :]
        obs_final1 = np.repeat(orig1[:, -1, :], repeats=T, axis=0)
        obs_final2 = np.repeat(orig2[:, -1, :], repeats=T, axis=0)

        model = self.load_model(type = "rotvec", state_dim = 7, action_dim = 7)

        with open("data/pot_states_rotvec_20.npy", "rb") as f:
            obs = np.load(f)

        planned_trajs = self.reactive_mpc_plan(model, [obs_init1[cond_idx], obs_init2[cond_idx]], [obs_final1[cond_idx], obs_final2[cond_idx]], obs[cond_idx], segment_length=H, total_steps=T, n_implement=1)
        planned_traj1 =  planned_trajs[0] * std + mean
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj1_%s.npy" % i, planned_traj1)
        planned_traj2 = planned_trajs[1] * std + mean
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj2_%s.npy" % i, planned_traj2)

        # Run the sampled trajectory in the environment
        for i in range(len(planned_traj1)):
            action = np.hstack([planned_traj1[i], planned_traj2[i]])
            obs, reward, done, info = self.env.step(action)

            if self.render:
                self.env.render()

    
        
if __name__ == "__main__":
    controller_config = load_composite_controller_config(robot="Kinova3", controller="utils/kinova.json")

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    cond_idx = 0
    player.get_demo(seed = cond_idx*10, cond_idx = cond_idx, H=25, T=250)
