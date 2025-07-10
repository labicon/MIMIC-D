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

    # def load_model(self, expert_data1, expert_data2, obs_init1, obs_init2, obs, state_dim = 7, action_dim = 7):
    #     model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
    #     H = 25 # horizon, length of each trajectory
    #     T = 250 # total time steps

    #     obs = np.repeat(obs, repeats=T, axis=0)
    #     obs1 = np.hstack([obs_init1, obs_init2, obs])
    #     obs2 = np.hstack([obs_init2, obs_init1, obs])
    #     obs1 = torch.FloatTensor(obs1).to(device)
    #     obs2 = torch.FloatTensor(obs2).to(device)
    #     attr1 = obs1
    #     attr2 = obs2
    #     attr_dim1 = attr1.shape[1]
    #     attr_dim2 = attr2.shape[1]

    #     # Preparing expert data for training
    #     actions1 = expert_data1[:, :H, :]
    #     actions2 = expert_data2[:, :H, :]
    #     actions1 = torch.FloatTensor(actions1).to(device)
    #     actions2 = torch.FloatTensor(actions2).to(device)
    #     sigma_data1 = actions1.std().item()
    #     sigma_data2 = actions2.std().item()

    #     env = TwoArmLift(state_size=state_dim, action_size=action_dim)

    #     # Load the model
    #     action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
    #     action_cond_ode.load(extra="_lift_mpc_P25E1_crosscond_nofinalpos_fullstate_nolf")

    #     return action_cond_ode


    def load_model(self, expert_data1, expert_data2, obs_init1, obs_init2, obs, state_dim = 7, action_dim = 7):
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 25 # horizon, length of each trajectory

        # The 'obs' parameter is already the correctly repeated pot states
        # The hstack will now work as all inputs are 2D with the same number of rows
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

        env = TwoArmLift(state_size=state_dim, action_size=action_dim)

        # Load the model
        action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
        action_cond_ode.load(extra="_lift_mpc_P25E1_crosscond_nofinalpos_fullstate_nolf")

        return action_cond_ode
    
    def obs_to_state(self, obs):
        """
        Read the two arms’ current 7D states (local pos, rotation-vector, gripper)
        but *convert* world EEF pos into the robot’s own base frame.
        """
        # --- robot 0 ---
        # 1) world→local for position
        world_pos0 = obs["robot0_eef_pos"]                # (3,)
        # subtract the base origin, then rotate back into base axes:
        local_pos0 = self.robot0_base_ori_rotm.T @ (world_pos0 - self.robot0_base_pos)
        # 2) rotation‐vector (you can leave this in world frame if your training used world rotvec,
        #    or you can similarly re‐express it in local axes—just be consistent with training!)
        quat0     = obs["robot0_eef_quat"]                # (4,) world quaternion
        rotvec0   = R.from_quat(quat0).as_rotvec()         # (3,)
        # 3) gripper joint position
        grip0     = self.env.sim.data.qpos[self.qpos_index_0]
        state0    = np.hstack([local_pos0, rotvec0, grip0])  # (7,)

        # --- robot 1 ---
        world_pos1 = obs["robot1_eef_pos"]
        local_pos1 = self.robot1_base_ori_rotm.T @ (world_pos1 - self.robot1_base_pos)
        quat1      = obs["robot1_eef_quat"]
        rotvec1    = R.from_quat(quat1).as_rotvec()
        grip1      = self.env.sim.data.qpos[self.qpos_index_1]
        state1     = np.hstack([local_pos1, rotvec1, grip1])

        return state0, state1
    
    
    # def reactive_mpc_plan(self, ode_model, initial_states, obs, segment_length=25, total_steps=250, n_implement=5):
    #     """
    #     Plans a full trajectory (total_steps long) by iteratively planning
    #     segment_length-steps using the diffusion model and replanning at every timestep.
        
    #     Parameters:
    #     - ode_model: the Conditional_ODE (diffusion model) instance.
    #     - env: your environment, which must implement reset_to() and step().
    #     - initial_state: a numpy array of shape (state_size,) (the current state).
    #     - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
    #     - model_i: the index of the agent/model being planned for.
    #     - segment_length: number of timesteps to plan in each segment.
    #     - total_steps: total length of the planned trajectory.
        
    #     Returns:
    #     - full_traj: a numpy array of shape (total_steps, state_size)
    #     """
    #     full_traj = []
    #     current_states = initial_states.copy()

    #     for seg in range(total_steps // n_implement):
    #         segments = []
    #         base_states = current_states.copy()
    #         for i in range(len(current_states)):
    #             cond = [base_states[i]]
    #             for j in range(len(current_states)):
    #                 if j != i:
    #                     cond.append(base_states[j])
    #             cond.append(obs)
    #             cond = np.hstack(cond)
    #             cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
    #             sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
    #             seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

    #             if seg == 0:
    #                 segments.append(seg_i[0:n_implement,:])
    #             else:
    #                 segments.append(seg_i[1:n_implement+1,:])
                
    #         for t in range(n_implement):
    #             action = np.hstack([segments[i][t] * self.std + self.mean for i in range(2)])
    #             obs_env, reward, done, info = self.env.step(action)
    #             if self.render:
    #                 self.env.render()
    #             # breakpoint()
    #             state1, state2 = self.obs_to_state(obs_env)
    #             # state1 = obs_env[:7]
    #             # state2 = obs_env[7:14]
    #             current_states = [(state1 - self.mean)/self.std, (state2 - self.mean)/self.std]

    #         seg_array = np.stack(segments, axis=0)
    #         full_traj.append(seg_array)

    #     full_traj = np.concatenate(full_traj, axis=1) 
    #     print("Full trajectory shape: ", np.shape(full_traj))
    #     return np.array(full_traj)
    

    def reactive_mpc_plan(self, ode_model, initial_states, obs, segment_length=25, total_steps=250, n_implement=1):
        """
        Plans a full trajectory by iteratively planning segments and executing them
        one step at a time, using online state feedback with correct normalization.
        """
        current_states = [s.copy() for s in initial_states]
        
        for t in range(total_steps):
            # --- Plan a short trajectory segment from the current NORMALIZED state ---
            segments = []
            for i in range(len(current_states)):
                cond = [current_states[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                cond.append(obs)
                cond = np.hstack(cond) #
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0) #
                
                sampled_segment = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i) #
                segments.append(sampled_segment.cpu().detach().numpy()[0])

            # --- DE-NORMALIZE the first action step using ACTION statistics ---
            action_t_robot0 = segments[0][0] * self.action_std + self.action_mean
            action_t_robot1 = segments[1][0] * self.action_std + self.action_mean
            action = np.hstack([action_t_robot0, action_t_robot1])

            # --- Step the environment ---
            obs_env, reward, done, info = self.env.step(action) #
            if self.render:
                self.env.render() #
            
            # --- Get the new RAW state from the simulator ---
            state0_new_raw, state1_new_raw = self.obs_to_state(obs_env) #

            # --- NORMALIZE the new state using STATE statistics for the next iteration ---
            current_states[0] = (state0_new_raw - self.state_mean) / self.state_std
            current_states[1] = (state1_new_raw - self.state_mean) / self.state_std

    
    # def get_demo(self, seed, cond_idx, H=25, T=250):
    #     """
    #     Main file to get the demonstration data
    #     """
    #     obs = self.reset(seed)

    #     # Loading
    #     expert_data = np.load("data/expert_actions_rotvec_20.npy")
    #     expert_data1 = expert_data[:, :, :7]
    #     expert_data2 = expert_data[:, :, 7:14]
    #     expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
    #     expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)
    #     combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
    #     self.mean = np.mean(combined_data, axis=(0,1))
    #     self.std = np.std(combined_data, axis=(0,1))

    #     # Normalize data
    #     expert_data1 = (expert_data1 - self.mean) / self.std
    #     expert_data2 = (expert_data2 - self.mean) / self.std

    #     obs_init1 = expert_data1[:, 0, :]
    #     obs_init2 = expert_data2[:, 0, :]

    #     with open("data/pot_states_rotvec_20.npy", "rb") as f:
    #         obs = np.load(f)

    #     model = self.load_model(expert_data1, expert_data2, obs_init1, obs_init2, obs, state_dim = 7, action_dim = 7)

    #     planned_trajs = self.reactive_mpc_plan(model, [obs_init1[cond_idx], obs_init2[cond_idx]], obs[cond_idx], segment_length=H, total_steps=T, n_implement=1)
    #     planned_traj1 =  planned_trajs[0] * self.std + self.mean
    #     # np.save("sampled_trajs/mpc_P34E5/mpc_traj1_%s.npy" % i, planned_traj1)
    #     planned_traj2 = planned_trajs[1] * self.std + self.mean
    #     # np.save("sampled_trajs/mpc_P34E5/mpc_traj2_%s.npy" % i, planned_traj2)

    #     # Run the sampled trajectory in the environment
    #     # for i in range(len(planned_traj1)):
    #     #     action = np.hstack([planned_traj1[i], planned_traj2[i]])
    #     #     obs, reward, done, info = self.env.step(action)

    #     #     if self.render:
    #     #         self.env.render()
    

    def get_demo(self, seed, cond_idx, H=25, T=250):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed)

        # --- ACTION STATISTICS ---
        expert_actions = np.load("data/expert_actions_rotvec_20.npy")
        expert_actions1 = expert_actions[:, :, :7]
        expert_actions2 = expert_actions[:, :, 7:14]
        combined_actions = np.concatenate((expert_actions1, expert_actions2), axis=0)
        self.action_mean = np.mean(combined_actions, axis=(0,1))
        self.action_std = np.std(combined_actions, axis=(0,1))

        # --- STATE STATISTICS ---
        expert_states = np.load("data/expert_states_rotvec_20.npy")
        expert_states1 = expert_states[:, :, :7]
        expert_states2 = expert_states[:, :, 7:14]
        demo_horizon = expert_states1.shape[1] # Get the full demo length (e.g., 250)
        combined_states = np.concatenate((expert_states1, expert_states2), axis=0)
        self.state_mean = np.mean(combined_states, axis=(0,1))
        self.state_std = np.std(combined_states, axis=(0,1))

        # --- Create MPC datasets for both ACTIONS and STATES ---
        expert_actions1_mpc = create_mpc_dataset(expert_actions1, planning_horizon=H)
        expert_actions2_mpc = create_mpc_dataset(expert_actions2, planning_horizon=H)
        expert_states1_mpc = create_mpc_dataset(expert_states1, planning_horizon=H)
        expert_states2_mpc = create_mpc_dataset(expert_states2, planning_horizon=H)

        # Normalize the action dataset for model loading
        expert_actions1_normalized = (expert_actions1_mpc - self.action_mean) / self.action_std
        expert_actions2_normalized = (expert_actions2_mpc - self.action_mean) / self.action_std

        # --- Use the first state of each MPC sub-trajectory as the initial condition ---
        obs_init1 = expert_states1_mpc[:, 0, :]
        obs_init2 = expert_states2_mpc[:, 0, :]
        
        # Normalize the initial states using state statistics
        obs_init1_normalized = (obs_init1 - self.state_mean) / self.state_std
        obs_init2_normalized = (obs_init2 - self.state_mean) / self.state_std

        # --- Load and repeat pot states to match the number of MPC samples ---
        with open("data/pot_states_rotvec_20.npy", "rb") as f:
            obs_pot = np.load(f)
        obs_pot_repeated = np.repeat(obs_pot, repeats=demo_horizon, axis=0)

        # --- Load the model with correctly shaped 2D arrays ---
        model = self.load_model(expert_actions1_normalized, expert_actions2_normalized,
                                obs_init1_normalized, obs_init2_normalized, obs_pot_repeated,
                                state_dim = 7, action_dim = 7)

        # Get the specific initial state for the desired demo run
        initial_state_for_run_1 = (expert_states1[cond_idx, 0, :] - self.state_mean) / self.state_std
        initial_state_for_run_2 = (expert_states2[cond_idx, 0, :] - self.state_mean) / self.state_std
        pot_state_for_run = obs_pot[cond_idx]

        self.reactive_mpc_plan(model, [initial_state_for_run_1, initial_state_for_run_2], pot_state_for_run,
                            segment_length=H, total_steps=T)

    
        
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
    cond_idx = 1
    player.get_demo(seed = cond_idx*10, cond_idx = cond_idx, H=25, T=250)
