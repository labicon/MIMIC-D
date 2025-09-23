# This script is used to sample the Conditional ODE model for the Two Arm Lift task and execute the demo.
# It uses the 3-dimensional rotation vector of the arm's state and action.

import time
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle as pkl
import copy
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from env import TwoArmLiftRole
from scipy.spatial.transform import Rotation as R\

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"

class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h

class BigImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=7, horizon=10,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon

        self.input = nn.Linear(input_size, hidden_size)

        # Learnable horizon/position embeddings (0..horizon-1)
        self.step_embed = nn.Embedding(horizon, hidden_size)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch_size, input_size)
        B = x.size(0)
        device = x.device

        # Encode input -> (B, hidden)
        h = self.input(x)  # (B, hidden_size)

        # Expand across horizon and add learnable step embeddings
        h = h.unsqueeze(1).expand(B, self.horizon, -1)              # (B, H, hidden)
        steps = torch.arange(self.horizon, device=device)           # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)                 # (B, H, hidden)

        # Per-step MLP blocks
        for blk in self.blocks:
            h = blk(h)                                              # (B, H, hidden)

        # Project to outputs per step
        out = self.head(h)                                          # (B, H, output_size)
        return out

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

    def load_model(self, expert_data1, expert_data2, obs_init1, obs_init2, obs, state_dim = 7, action_dim = 7):
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 25 # horizon, length of each trajectory
        T = 700 # total time steps

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

        env = TwoArmLift(state_size=state_dim, action_size=action_dim)

        model1 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
        model2 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)

        return action_cond_ode

    
    def obs_to_state(self, obs):
        """
        Read the two arms’ current 7D states (local pos, rotation-vector, gripper)
        using the correct end-effector SITE frame for rotation.
        """

        # --- Robot 0 ---
        # 1) World->local for position
        world_pos0 = obs["robot0_eef_pos"]
        local_pos0 = self.robot0_base_ori_rotm.T @ (world_pos0 - self.robot0_base_pos)
        
        # 2) Get rotation vector from the SITE quaternion
        quat0_site = obs["robot0_eef_quat_site"]
        R_world_to_site0 = R.from_quat(quat0_site).as_matrix()
        R_base_to_site0 = self.robot0_base_ori_rotm.T @ R_world_to_site0
        rotvec0 = R.from_matrix(R_base_to_site0).as_rotvec()

        # 3) Gripper joint position
        grip0 = self.env.sim.data.qpos[self.qpos_index_0]
        state0 = np.hstack([local_pos0, rotvec0, grip0])

        # --- Robot 1 ---
        # 1) World->local for position
        world_pos1 = obs["robot1_eef_pos"]
        local_pos1 = self.robot1_base_ori_rotm.T @ (world_pos1 - self.robot1_base_pos)

        # 2) Get rotation vector from the SITE quaternion
        quat1_site = obs["robot1_eef_quat_site"]
        R_world_to_site1 = R.from_quat(quat1_site).as_matrix()
        R_base_to_site1 = self.robot1_base_ori_rotm.T @ R_world_to_site1
        rotvec1 = R.from_matrix(R_base_to_site1).as_rotvec()

        # 3) Gripper joint position
        grip1 = self.env.sim.data.qpos[self.qpos_index_1]
        state1 = np.hstack([local_pos1, rotvec1, grip1])

        return state0, state1
    
    
    def reactive_mpc_plan(self, models, initial_states, pot, segment_length=25, total_steps=325, n_implement=2):
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

            # 1) sample a full normalized‐action segment for each arm
            for i in range(len(base_states)):
                # build conditioning vector exactly as in training
                cond = [base_states[i]] + [base_states[j] for j in range(len(base_states)) if j!=i] + [pot]
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)

                # sampled = ode_model.sample(
                #     attr=cond_tensor,
                #     traj_len=segment_length,
                #     n_samples=1,
                #     w=1.,
                #     model_index=i
                # )
                sampled = models[i](cond_tensor)  # (1, segment_length, 7)
                seg_i = sampled[0].cpu().detach().numpy()  # (segment_length, 7)

                # pick out the n_implement steps and the “next” state to re‐condition on
                if seg == 0:
                    step_block    = seg_i[0:n_implement, :]        # shape (n_implement,7)
                    next_norm     = seg_i[n_implement-1, :]        # shape (7,)
                else:
                    step_block    = seg_i[1:n_implement+1, :]
                    next_norm     = seg_i[n_implement, :]

                segments.append(step_block)
                current_states[i] = next_norm

            # 2) execute those n_implement actions on the real robot
            for t in range(n_implement):
                action1 = segments[0][t] * self.std_arm1 + self.mean_arm1
                action2 = segments[1][t] * self.std_arm2 + self.mean_arm2
                action = np.hstack([action1, action2])
                obs_env, reward, done, info = self.env.step(action)
                if self.render:
                    self.env.render()

                # 3) (optional) re‐condition on true state instead of predicted:
                state1, state2 = self.obs_to_state(obs_env)
                current_states = [
                    (state1 - self.mean_arm1)/self.std_arm1,
                    (state2 - self.mean_arm2)/self.std_arm2,
                ]

            full_traj.append(np.stack([s for s in segments], axis=0))

        full_traj = np.concatenate(full_traj, axis=1)

        print("Full trajectory shape: ", np.shape(full_traj))
        return np.array(full_traj)

    
    def get_demo(self, seed, cond_idx, H=25, T=700):
        """
        Main file to get the demonstration data
        """
        obs = self.reset(seed)

        # Load expert data
        expert_data = np.load("data/expert_actions_newslower_20.npy")
        expert_data1 = expert_data[:, :, :7]
        expert_data2 = expert_data[:, :, 7:14]

        # eps = 1e-8
        # self.mean_arm1 = expert_data1.mean(axis=(0, 1))
        # self.std_arm1  = expert_data1.std(axis=(0, 1)) + eps
        # self.mean_arm2 = expert_data2.mean(axis=(0, 1))
        # self.std_arm2  = expert_data2.std(axis=(0, 1)) + eps


        expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
        expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

        # Compute mean and standard deviation
        eps = 1e-8
        self.mean_arm1 = expert_data1.mean(axis=(0, 1))
        self.std_arm1  = expert_data1.std(axis=(0, 1)) + eps
        self.mean_arm2 = expert_data2.mean(axis=(0, 1))
        self.std_arm2  = expert_data2.std(axis=(0, 1)) + eps

        # Normalize data
        expert_data1 = (expert_data1 - self.mean_arm1) / self.std_arm1
        expert_data2 = (expert_data2 - self.mean_arm2) / self.std_arm2

        # Preparing expert data for training
        actions1 = expert_data1[:, :H, :]
        actions2 = expert_data2[:, :H, :]
        actions1 = torch.FloatTensor(actions1).to(device)
        actions2 = torch.FloatTensor(actions2).to(device)
        sigma_data1 = actions1.std().item()
        sigma_data2 = actions2.std().item()

        with open("data/pot_start_newslower_20.npy", "rb") as f:
            pot = np.load(f)

        pot_mean = np.mean(pot, axis=0)
        pot_std = np.std(pot, axis=0)
        pot = (pot - pot_mean) / pot_std
        pot_dim = pot.shape[1]

        obs = np.repeat(pot, repeats=T, axis=0)

        obs_init1 = expert_data1[:, 0, :]
        obs_init2 = expert_data2[:, 0, :]
        obs1 = np.hstack([obs_init1, obs_init2, obs])
        obs2 = np.hstack([obs_init2, obs_init1, obs])
        obs1 = torch.FloatTensor(obs1).to(device)
        obs2 = torch.FloatTensor(obs2).to(device)
        attr1 = obs1
        attr2 = obs2
        attr_dim1 = attr1.shape[1]
        attr_dim2 = attr2.shape[1]

        input_size = attr1.shape[1]

        # import ipdb; ipdb.set_trace()

        bc1 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
        bc2 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
        save_dir = "trained_models/bc"
        bc1.load_state_dict(torch.load(os.path.join(save_dir, "model1_big.pth"), map_location=device))
        bc2.load_state_dict(torch.load(os.path.join(save_dir, "model2_big.pth"), map_location=device))
        bc1.eval(); bc2.eval()
        bc_models = [bc1, bc2]
        planned_trajs = self.reactive_mpc_plan(bc_models, [obs_init1[cond_idx], obs_init2[cond_idx]], obs[cond_idx], segment_length=H, total_steps=T*2, n_implement=10)
        planned_traj1 =  planned_trajs[0] * self.std_arm1 + self.mean_arm1
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj1_%s.npy" % i, planned_traj1)
        planned_traj2 = planned_trajs[1] * self.std_arm2 + self.mean_arm2
        # np.save("sampled_trajs/mpc_P34E5/mpc_traj2_%s.npy" % i, planned_traj2)

    
        
if __name__ == "__main__":
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    T = 700
    H = 25

    env = TwoArmLiftRole(
    robots=["Kinova3", "Kinova3"],
    gripper_types="default",
    horizon=T*2,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    render_camera=None,
    )

    player = PolicyPlayer(env, render = False)
    cond_idx = 0
    player.get_demo(seed = cond_idx*10, cond_idx = cond_idx, H=H, T=T)
