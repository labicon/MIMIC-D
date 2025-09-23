# magail_sampler.py
import os
import math
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from robosuite.controllers import load_composite_controller_config
from env import TwoArmLiftRole
import argparse

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Helpers ----
def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon
    result = []
    for traj in expert_data:
        for start_idx in range(n_subtraj):
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                sub_traj = traj[start_idx:]
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)
    result = np.stack(result, axis=0)
    return result

# ---- Model architecture (matches training) ----
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

class SeqGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=256, action_dim=7, horizon=25,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Linear(input_size, hidden_size)
        self.step_embed = nn.Embedding(horizon, hidden_size)
        self.blocks = nn.ModuleList([MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
                                     for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, attrs):  # attrs: (B, input_size)
        B = attrs.size(0)
        dev = attrs.device
        h = self.inp(attrs)                                # (B, hidden)
        h = h.unsqueeze(1).expand(B, self.horizon, -1)     # (B, H, hidden)
        steps = torch.arange(self.horizon, device=dev)     # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)        # (B, H, hidden)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)                                 # (B, H, action_dim)
        return out

# ---- Policy player (MAGAIL sequence generator version) ----
class PolicyPlayer:
    def __init__(self, env, render=False, h=25, model_dir="trained_models/magail_seq"):
        self.env = env
        self.render = render
        self.device = device
        self.H = h
        self.model_dir = model_dir

        # placeholders for normalization stats -- set in get_demo()
        self.mean_arm1 = None; self.std_arm1 = None
        self.mean_arm2 = None; self.std_arm2 = None
        self.pot_mean = None; self.pot_std = None

        # base/robot transforms for obs_to_state (set in reset)
        self.robot0_base_pos = None
        self.robot1_base_pos = None
        self.robot0_base_ori_rotm = None
        self.robot1_base_ori_rotm = None
        self.qpos_index_0 = None
        self.qpos_index_1 = None

        # models
        self.G1 = None
        self.G2 = None

    def reset(self, seed=0):
        np.random.seed(seed)
        obs = self.env.reset()

        # set base frames & joint indices (assume names match)
        try:
            robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
            robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
            self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
            self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
            self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))
            self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3,3))
        except Exception:
            # Not fatal — some env variants may store differently. We'll still use obs fields in obs_to_state.
            pass

        # find gripper joint qpos addresses (names from your BC code)
        try:
            jnt_id_0 = self.env.sim.model.joint_name2id("gripper0_right_finger_joint")
            jnt_id_1 = self.env.sim.model.joint_name2id("gripper1_right_finger_joint")
            self.qpos_index_0 = self.env.sim.model.jnt_qposadr[jnt_id_0]
            self.qpos_index_1 = self.env.sim.model.jnt_qposadr[jnt_id_1]
        except Exception:
            # fallback - some envs use different naming; obs_to_state extracts gripper from obs
            self.qpos_index_0 = None
            self.qpos_index_1 = None

        return obs

    def obs_to_state(self, obs):
        """
        Convert environment observation dict -> (state0, state1) each 7-d:
        [local_pos(3), rotvec(3), gripper(1)]
        This follows your BC code's choice of SITE quaternion -> rotvec and transforms to base frame.
        """
        # Robot 0
        world_pos0 = obs.get("robot0_eef_pos", None)
        if world_pos0 is None:
            # fallback names
            world_pos0 = obs["eef_pos_0"] if "eef_pos_0" in obs else obs["robot0_eef_pos"]

        quat0_site = obs.get("robot0_eef_quat_site", obs.get("robot0_eef_quat", None))
        if quat0_site is None:
            # fallback, try to parse from obs fields
            quat0_site = obs.get("robot0_eef_quat_site", [1,0,0,0])

        # Get local pos in robot base frame if base info exists
        if self.robot0_base_ori_rotm is not None and self.robot0_base_pos is not None:
            local_pos0 = self.robot0_base_ori_rotm.T @ (world_pos0 - self.robot0_base_pos)
        else:
            local_pos0 = np.array(world_pos0)

        R_world_to_site0 = R.from_quat(quat0_site).as_matrix()
        if self.robot0_base_ori_rotm is not None:
            R_base_to_site0 = self.robot0_base_ori_rotm.T @ R_world_to_site0
        else:
            R_base_to_site0 = R_world_to_site0
        rotvec0 = R.from_matrix(R_base_to_site0).as_rotvec()

        if self.qpos_index_0 is not None:
            grip0 = self.env.sim.data.qpos[self.qpos_index_0]
        else:
            # try to read from obs
            grip0 = float(obs.get("robot0_gripper_qpos", [0.0])[0])

        state0 = np.hstack([local_pos0, rotvec0, grip0])

        # Robot 1
        world_pos1 = obs.get("robot1_eef_pos", None)
        if world_pos1 is None:
            world_pos1 = obs["eef_pos_1"] if "eef_pos_1" in obs else obs["robot1_eef_pos"]

        quat1_site = obs.get("robot1_eef_quat_site", obs.get("robot1_eef_quat", None))
        if quat1_site is None:
            quat1_site = [1,0,0,0]

        if self.robot1_base_ori_rotm is not None and self.robot1_base_pos is not None:
            local_pos1 = self.robot1_base_ori_rotm.T @ (world_pos1 - self.robot1_base_pos)
        else:
            local_pos1 = np.array(world_pos1)

        R_world_to_site1 = R.from_quat(quat1_site).as_matrix()
        if self.robot1_base_ori_rotm is not None:
            R_base_to_site1 = self.robot1_base_ori_rotm.T @ R_world_to_site1
        else:
            R_base_to_site1 = R_world_to_site1
        rotvec1 = R.from_matrix(R_base_to_site1).as_rotvec()

        if self.qpos_index_1 is not None:
            grip1 = self.env.sim.data.qpos[self.qpos_index_1]
        else:
            grip1 = float(obs.get("robot1_gripper_qpos", [0.0])[0])

        state1 = np.hstack([local_pos1, rotvec1, grip1])

        return state0, state1

    def load_models(self, input_size, hidden_size=256, action_dim=7, horizon=25):
        # instantiate SeqGenerators with matching architecture and load weights
        self.G1 = SeqGenerator(input_size=input_size, hidden_size=hidden_size,
                               action_dim=action_dim, horizon=horizon).to(self.device)
        self.G2 = SeqGenerator(input_size=input_size, hidden_size=hidden_size,
                               action_dim=action_dim, horizon=horizon).to(self.device)

        g1_path = os.path.join(self.model_dir, "G1.pth")
        g2_path = os.path.join(self.model_dir, "G2.pth")
        assert os.path.exists(g1_path) and os.path.exists(g2_path), f"Missing models in {self.model_dir}"

        self.G1.load_state_dict(torch.load(g1_path, map_location=self.device))
        self.G2.load_state_dict(torch.load(g2_path, map_location=self.device))
        self.G1.eval(); self.G2.eval()

    def reactive_mpc_plan(self, models, initial_states, pot_raw, segment_length=25,
                          total_steps=700, n_implement=10):
        """
        models: [G1, G2] sequence generators (torch models)
        initial_states: list of two normalized starting states (each 7-d)
        pot_raw: raw pot init vector (unnormalized)
        returns: full_traj array shaped (2, total_steps, 7) (denormalized actions)
        """
        G1, G2 = models
        pot_norm = (pot_raw - self.pot_mean) / self.pot_std

        current_states = [initial_states[0].astype(np.float32).copy(), initial_states[1].astype(np.float32).copy()]
        full_traj = []

        n_impl = int(max(1, min(n_implement, segment_length)))
        n_iters = int(math.ceil(total_steps / n_impl))

        for seg in range(n_iters):
            segments = []
            base_states = [s.copy() for s in current_states]

            # plan per arm
            for i, G in enumerate([G1, G2]):
                cond = np.hstack([base_states[i], base_states[1-i], pot_norm]).astype(np.float32)
                cond_t = torch.tensor(cond, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, input_size)
                with torch.no_grad():
                    seq = G(cond_t)   # (1, H, 7)
                seq_np = seq.squeeze(0).cpu().numpy()  # (H, 7)

                # pick executed block & next provisional state
                if seg == 0:
                    step_block = seq_np[0:n_impl, :]
                    next_norm = seq_np[n_impl - 1, :]
                else:
                    # shift by one to mimic the BC code's reconditioning scheme
                    step_block = seq_np[1:n_impl+1, :]
                    next_norm = seq_np[n_impl, :]

                segments.append(step_block)
                current_states[i] = next_norm

            # execute on the real env for n_impl steps (recondition after each step)
            for t in range(n_impl):
                action1 = segments[0][t] * self.std_arm1 + self.mean_arm1
                action2 = segments[1][t] * self.std_arm2 + self.mean_arm2
                action = np.hstack([action1, action2]).astype(np.float32)

                obs_env, reward, done, info = self.env.step(action)
                if self.render:
                    self.env.render()

                # recondition using true observed state
                state1, state2 = self.obs_to_state(obs_env)
                current_states = [
                    (state1 - self.mean_arm1) / self.std_arm1,
                    (state2 - self.mean_arm2) / self.std_arm2
                ]

                # store executed (denormalized) actions
                full_traj.append(np.stack([action1, action2], axis=0))  # shape (2,)

                if done:
                    break

            if done:
                break

        if len(full_traj) == 0:
            return np.zeros((2, 0, 7), dtype=np.float32)

        full_traj = np.stack(full_traj, axis=1)  # (2, total_executed_steps, 7)
        print("Full trajectory shape:", full_traj.shape)
        return full_traj

    def get_demo(self, seed, cond_idx, H=25, T=700, n_implement=10):
        """
        Wrapper that:
          - resets env
          - computes normalization stats from expert dataset (same as training)
          - loads models
          - prepares initial conditioning and calls reactive_mpc_plan
        """
        # reset env (sets base frames, indices)
        obs = self.reset(seed)

        # load expert data to compute normalization stats (must match training)
        expert = np.load("data/expert_actions_newslower_20.npy")  # (N_traj, full_T, 14)
        arm1 = expert[:, :, :7]
        arm2 = expert[:, :, 7:14]

        # compute statistics exactly like training
        arm1_w = create_mpc_dataset(arm1, planning_horizon=H)
        arm2_w = create_mpc_dataset(arm2, planning_horizon=H)
        eps = 1e-8
        self.mean_arm1 = arm1_w.mean(axis=(0, 1)); self.std_arm1 = arm1_w.std(axis=(0, 1)) + eps
        self.mean_arm2 = arm2_w.mean(axis=(0, 1)); self.std_arm2 = arm2_w.std(axis=(0, 1)) + eps

        # load pot stats and per-window pot records (same as training)
        pot_all = np.load("data/pot_start_newslower_20.npy")  # (n_traj, pot_dim)
        self.pot_mean = pot_all.mean(axis=0); self.pot_std = pot_all.std(axis=0) + eps
        pot_dim = pot_all.shape[1]

        # prepare per-window attributes (same tiling as training)
        N = arm1_w.shape[0]  # number of windows
        N0 = pot_all.shape[0]
        repeat_factor = N // N0
        pot_tiled = np.repeat(pot_all, repeats=repeat_factor, axis=0)  # (N, pot_dim)

        obs_init1 = arm1_w[:, 0, :]    # (N, 7)  - raw values (not normalized)
        obs_init2 = arm2_w[:, 0, :]

        # Insert live state for the chosen cond_idx (like your BC code)
        # get live obs
        obs_live = self.reset(seed)
        state1_live, state2_live = self.obs_to_state(obs_live)
        init1_norm = (state1_live - self.mean_arm1) / self.std_arm1
        init2_norm = (state2_live - self.mean_arm2) / self.std_arm2

        total_windows = obs_init1.shape[0]
        cond_idx = cond_idx % total_windows
        # Overwrite only the chosen window's init with the live normalized states
        obs_init1[cond_idx] = init1_norm
        obs_init2[cond_idx] = init2_norm

        # Build attributes as in training (but we'll pass only the chosen cond to the models during MPC)
        obs_horizon = np.repeat(pot_tiled, repeats=H, axis=0)[:N]  # shape (N, pot_dim) repeated across H not needed beyond attr construction

        # Now prepare input_size (init1, init2, pot)
        input_size = obs_init1.shape[1] + obs_init2.shape[1] + pot_dim  # although obs_init1 already size 7, we will form cond below

        # Load models (ensure models expect input_size = 7 + 7 + pot_dim)
        self.load_models(input_size=input_size, horizon=H)

        # Choose pot raw vector for the trajectory's trajectory index
        horizon = arm1.shape[1]
        traj_idx = cond_idx // horizon
        pot0_raw = pot_all[traj_idx]

        # prepare normalized initial states for the starting window
        init1_raw = obs_init1[cond_idx]   # already normalized (we overwrote with live norm)
        init2_raw = obs_init2[cond_idx]

        # call reactive MPC: models list, initial states (already normalized), pot raw
        planned = self.reactive_mpc_plan([self.G1, self.G2],
                                         initial_states=[init1_raw, init2_raw],
                                         pot_raw=pot0_raw,
                                         segment_length=H,
                                         total_steps=T*2,
                                         n_implement=n_implement)

        # planned returned is denormalized actions, shape (2, executed_steps, 7)
        planned_traj1 = planned[0]  # (executed_steps, 7)
        planned_traj2 = planned[1]
        return planned_traj1, planned_traj2


# ---- Main runner ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_idx", type=int, default=0, help="conditioning window index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model_dir", type=str, default="trained_models/magail_seq")
    parser.add_argument("--T", type=int, default=700)
    parser.add_argument("--H", type=int, default=25)
    parser.add_argument("--n_impl", type=int, default=10)
    args = parser.parse_args()

    # create env (match your BC code call)
    controller_config = load_composite_controller_config(robot="Kinova3", controller="kinova.json")

    env = TwoArmLiftRole(
        robots=["Kinova3", "Kinova3"],
        gripper_types="default",
        horizon=args.T * 2,
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        render_camera=None,
        # has_renderer=args.render,
        # has_offscreen_renderer=not args.render,
        # use_camera_obs=False,
        # render_camera=None,
    )

    player = PolicyPlayer(env, render=args.render, h=args.H, model_dir=args.model_dir)
    planned1, planned2 = player.get_demo(seed=args.seed, cond_idx=args.cond_idx, H=args.H, T=args.T, n_implement=args.n_impl)

    print("Planned traj 1 shape:", planned1.shape)
    print("Planned traj 2 shape:", planned2.shape)

    # Save example outputs
    out_dir = "sampled_trajs_magail_seq"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"planned_traj1_cond{args.cond_idx}_seed{args.seed}.npy"), planned1)
    np.save(os.path.join(out_dir, f"planned_traj2_cond{args.cond_idx}_seed{args.seed}.npy"), planned2)
    print("Saved planned trajectories to", out_dir)