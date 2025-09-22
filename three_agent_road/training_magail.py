# training_magail_ctde_v2.py

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1) Device & performance tweak
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 2) Hyperparameters
state_dim   = 6       # e.g. [x, y, other_agent1_x, other_agent1_y, other_agent2_x, other_agent2_y]
action_dim  = 2       # e.g. next [x, y]
hidden_size = 64
batch_size  = 64
n_epochs    = 2000
lr_G        = 1e-4
lr_D        = 1e-4
GEN_WIDTH  = 2000
DISC_WIDTH, DISC_DEPTH = 768, 8
def count_params(m): return sum(p.numel() for p in m.parameters())

# 3) Models
class GenNet(nn.Module):
    def __init__(self, width=GEN_WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,   width), nn.ReLU(inplace=True),
            nn.Linear(width,       width), nn.ReLU(inplace=True),
            nn.Linear(width,       action_dim)
        )
    def forward(self, x): return self.net(x)

class DiscNet(nn.Module):
    def __init__(self, width=DISC_WIDTH, depth=DISC_DEPTH, spectral=True):
        super().__init__()
        in_dim = state_dim + action_dim  # 4 + 2 = 6
        def lin(i, o):
            layer = nn.Linear(i, o)
            return nn.utils.spectral_norm(layer) if spectral else layer
        layers = [lin(in_dim, width), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(depth - 1):
            layers += [lin(width, width), nn.LeakyReLU(0.2, inplace=True)]
        layers += [lin(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1)).squeeze(1)


G1 = GenNet(width=GEN_WIDTH).to(device)
G2 = GenNet(width=GEN_WIDTH).to(device)
G3 = GenNet(width=GEN_WIDTH).to(device)
D  = DiscNet(width=DISC_WIDTH, depth=DISC_DEPTH, spectral=True).to(device)
print(f"G total params: {count_params(G1)+count_params(G2)+count_params(G3):,}")
print(f"D params:       {count_params(D):,}")

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()) + list(G3.parameters()), lr=1e-4, betas=(0.5, 0.999))
optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce_logits = nn.BCEWithLogitsLoss()

# 4) Load expert data & build (s, a) pairs
datas = []
for idx in range(1, 4):
    datas.append(np.load(f'data/expert_data{idx}_400_traj_06_noise.npy'))

def build_sa(expert, other1, other2):
    S, A = [], []
    for i in range(len(expert)):
        traj = expert[i]
        other_traj1 = other1[i]
        other_traj2 = other2[i]
        for t in range(len(traj) - 1):
            s = np.hstack([traj[t], other_traj1[t], other_traj2[t]])  # (state_dim,)
            a = traj[t+1]                                            # (action_dim,)
            S.append(s); A.append(a)
    return np.stack(S), np.stack(A)

S1, A1 = build_sa(datas[0], datas[1], datas[2])
S2, A2 = build_sa(datas[1], datas[0], datas[2])
S3, A3 = build_sa(datas[2], datas[0], datas[1])

# 5) Joint normalization for states & actions
state_data  = np.concatenate([S1, S2, S3], axis=0)
action_data = np.concatenate([A1, A2, A3], axis=0)
state_mean, state_std   = state_data.mean(0),  state_data.std(0)  + 1e-6
action_mean, action_std = action_data.mean(0), action_data.std(0) + 1e-6
S1 = (S1 - state_mean) / state_std
S2 = (S2 - state_mean) / state_std
S3 = (S3 - state_mean) / state_std
A1 = (A1 - action_mean) / action_std
A2 = (A2 - action_mean) / action_std
A3 = (A3 - action_mean) / action_std

# 6) Torch datasets & loaders
tS1, tA1 = torch.tensor(S1, dtype=torch.float32), torch.tensor(A1, dtype=torch.float32)
tS2, tA2 = torch.tensor(S2, dtype=torch.float32), torch.tensor(A2, dtype=torch.float32)
tS3, tA3 = torch.tensor(S3, dtype=torch.float32), torch.tensor(A3, dtype=torch.float32)
loader1 = DataLoader(TensorDataset(tS1, tA1), batch_size=batch_size, shuffle=True, drop_last=True)
loader2 = DataLoader(TensorDataset(tS2, tA2), batch_size=batch_size, shuffle=True, drop_last=True)
loader3 = DataLoader(TensorDataset(tS3, tA3), batch_size=batch_size, shuffle=True, drop_last=True)

# 7) Training loop (centralized training structure)
# for epoch in range(1, n_epochs + 1):
#     G1.train(); G2.train(); G3.train(); D.train()
#     lossD_sum = 0.0
#     lossG_sum = 0.0

#     for (x1, a1), (x2, a2), (x3, a3) in zip(loader1, loader2, loader3):
#         x1, a1 = x1.to(device), a1.to(device)
#         x2, a2 = x2.to(device), a2.to(device)
#         x3, a3 = x3.to(device), a3.to(device)

#         # —— Discriminator update across all agents ——
#         with torch.no_grad():
#             fake1 = G1(x1)
#             fake2 = G2(x2)
#             fake3 = G3(x3)
#         real_logit1 = D(x1, a1)
#         fake_logit1 = D(x1, fake1)
#         real_logit2 = D(x2, a2)
#         fake_logit2 = D(x2, fake2)
#         real_logit3 = D(x3, a3)
#         fake_logit3 = D(x3, fake3)
#         lossD1 = 0.5*(bce_logits(real_logit1, torch.ones_like(real_logit1)) + bce_logits(fake_logit1, torch.zeros_like(fake_logit1)))
#         lossD2 = 0.5*(bce_logits(real_logit2, torch.ones_like(real_logit2)) + bce_logits(fake_logit2, torch.zeros_like(fake_logit2)))
#         lossD3 = 0.5*(bce_logits(real_logit3, torch.ones_like(real_logit3)) + bce_logits(fake_logit3, torch.zeros_like(fake_logit3)))
#         lossD = lossD1 + lossD2 + lossD3
#         optD.zero_grad()
#         lossD.backward()
#         optD.step()

#         # —— Generator update for each agent ——
#         fake1 = G1(x1)
#         fake2 = G2(x2)
#         fake3 = G3(x3)
#         lossG1 = bce_logits(D(x1, fake1), torch.ones_like(fake1[:, 0]))
#         lossG2 = bce_logits(D(x2, fake2), torch.ones_like(fake2[:, 0]))
#         lossG3 = bce_logits(D(x3, fake3), torch.ones_like(fake3[:, 0]))
#         lossG = lossG1 + lossG2 + lossG3
#         optG.zero_grad()
#         lossG.backward()
#         optG.step()

#         lossD_sum += lossD.item()
#         lossG_sum += lossG.item()

#     if epoch % 100 == 0:
#         avgD = lossD_sum / len(loader1)
#         avgG = lossG_sum / len(loader1)
#         print(f"Epoch {epoch:4d} | lossD: {avgD:.4f} | lossG: {avgG:.4f}")

# 8) Save models
save_path_G1 = "trained_models/magail_ctde/G1_nofinalpos_big.pth"
save_path_G2 = "trained_models/magail_ctde/G2_nofinalpos_big.pth"
save_path_G3 = "trained_models/magail_ctde/G3_nofinalpos_big.pth"
save_path_D  = "trained_models/magail_ctde/D_nofinalpos_big.pth"
# torch.save(G1.state_dict(), save_path_G1)
# torch.save(G2.state_dict(), save_path_G2)
# torch.save(G3.state_dict(), save_path_G3)
# torch.save(D.state_dict(),  save_path_D)
print("CTDE MAGAIL training complete; models saved under trained_models/magail_ctde/")

G1 = GenNet(width=GEN_WIDTH).to(device)
G2 = GenNet(width=GEN_WIDTH).to(device)
G3 = GenNet(width=GEN_WIDTH).to(device)
G1.load_state_dict(torch.load(save_path_G1, map_location=device))
G2.load_state_dict(torch.load(save_path_G2, map_location=device))
G3.load_state_dict(torch.load(save_path_G3, map_location=device))
G1.eval(); G2.eval(); G3.eval()

expert_data1 = np.load('data/expert_data1_400_traj_06_noise.npy')  # (n_traj, horizon, 2)
expert_data2 = np.load('data/expert_data2_400_traj_06_noise.npy')
expert_data3 = np.load('data/expert_data3_400_traj_06_noise.npy')

# def sample_trajectory(G, init_pos, init2, init3, horizon):
#     pos = init_pos.copy()
#     traj = [pos.copy()]
#     for _ in range(horizon-1):
#         s = np.hstack([pos, init2, init3])
#         s_norm = (s - state_mean) / state_std
#         s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
#         with torch.no_grad():
#             a_norm = G(s_t).cpu().numpy()[0]
#         a = a_norm * action_std + action_mean
#         pos = a.copy()
#         traj.append(pos.copy())
#     return np.stack(traj)
@torch.no_grad()
def rollout_two_agents(G1, G2, G3, init1, init2, init3, T, state_mean, state_std, action_mean, action_std, device):
    cur1 = init1.copy(); cur2 = init2.copy(); cur3 = init3.copy()
    traj1 = [cur1.copy()]; traj2 = [cur2.copy()]; traj3 = [cur3.copy()]
    for _ in range(T-1):
        s1 = np.hstack([cur1, cur2, cur3])  # other = current, not initial
        s2 = np.hstack([cur2, cur1, cur3])
        s3 = np.hstack([cur3, cur1, cur2])
        s1n = (s1 - state_mean) / state_std
        s2n = (s2 - state_mean) / state_std
        s3n = (s3 - state_mean) / state_std
        a1n = G1(torch.tensor(s1n, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
        a2n = G2(torch.tensor(s2n, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
        a3n = G3(torch.tensor(s3n, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
        next1 = a1n * action_std + action_mean     # absolute next state
        next2 = a2n * action_std + action_mean
        next3 = a3n * action_std + action_mean
        traj1.append(next1.copy()); traj2.append(next2.copy()); traj3.append(next3.copy())
        cur1, cur2, cur3 = next1, next2, next3
    return np.stack(traj1), np.stack(traj2), np.stack(traj3)

# 7) Sample & plot on single axes
initial_point_1 = np.array([0.0, 2.0])
final_point_1 = np.array([2.0, 0.0])
initial_point_2 = np.array([0.75, -2.0])
final_point_2 = np.array([0.75, 2.0])
initial_point_3 = np.array([-0.25, 0.75])
final_point_3 = np.array([1.75, 0.75])

num_samples = 100
noise_std = 0.6
threshold = 0.75
T = expert_data1.shape[1]

# plt.figure(figsize=(8, 8))
for s in range(10):
    seed = s * 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    path_vary = f"sampled_trajs/magail_seed{seed}/vary_init"
    path_static = f"sampled_trajs/magail_seed{seed}/static_init"
    os.makedirs(path_vary, exist_ok=True)
    os.makedirs(path_static, exist_ok=True)

    # plt.figure(figsize=(8, 8))
    for i in range(num_samples):

        while True:
            initial1 = initial_point_1 + np.random.uniform(-noise_std, noise_std, size=(2,))    
            initial2 = initial_point_2 + np.random.uniform(-noise_std, noise_std, size=(2,))
            initial3 = initial_point_3 + np.random.uniform(-noise_std, noise_std, size=(2,))

            d_init12 = np.linalg.norm(initial1 - initial2)
            d_init13 = np.linalg.norm(initial1 - initial3)
            d_init23 = np.linalg.norm(initial2 - initial3)

            if (d_init12 > threshold and d_init13 > threshold and d_init23 > threshold):
                break

        traj1, traj2, traj3 = rollout_two_agents(G1, G2, G3, initial1, initial2, initial3, T, state_mean, state_std, action_mean, action_std, device)

        np.save(os.path.join(path_vary, f"mpc_traj1_{i}.npy"), traj1)
        np.save(os.path.join(path_vary, f"mpc_traj2_{i}.npy"), traj2)
        np.save(os.path.join(path_vary, f"mpc_traj3_{i}.npy"), traj3)

        # traj1 = sample_trajectory(G1, init1, init2, init3, T)
        # np.save(f"sampled_trajs/magail_nofinalpos_big/vary_init/traj1_{i}.npy", traj1)
        # traj2 = sample_trajectory(G2, init2, init1, init3, T)c
        # np.save(f"sampled_trajs/magail_nofinalpos_big/vary_init/traj2_{i}.npy", traj2)
        # traj3 = sample_trajectory(G3, init3, init1, init2, T)
        # np.save(f"sampled_trajs/magail_nofinalpos_big/vary_init/traj3_{i}.npy", traj3)

        # plt.plot(traj1[:,0], traj1[:,1], color='blue', alpha=0.7)
        # plt.plot(traj2[:,0], traj2[:,1], color='orange', alpha=0.7)
        # plt.plot(traj3[:,0], traj3[:,1], color='green', alpha=0.7)

    # plt.title("MAGAIL‐Generated Trajectories")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # --- pick one fixed context ---------------------
    idx      = 0                                # choose any trajectory index
    init1    = expert_data1[idx, 0];  goal1 = expert_data1[idx, -1]
    init2    = expert_data2[idx, 0];  goal2 = expert_data2[idx, -1]
    init3    = expert_data3[idx, 0];  goal3 = expert_data3[idx, -1]
    T        = expert_data1.shape[1]

    # --- sample N rollouts for each agent ----------
    N = 100
    rollouts1, rollouts2, rollouts3 = [], [], []
    for _ in range(N):
        traj1, traj2, traj3 = rollout_two_agents(G1, G2, G3, init1, init2, init3, T, state_mean, state_std, action_mean, action_std, device)
        rollouts1.append(traj1)
        rollouts2.append(traj2)
        rollouts3.append(traj3)
    # rollouts1 = [sample_trajectory(G1, init1, init2, init3, T) for _ in range(N)]
    # rollouts2 = [sample_trajectory(G2, init2, init1, init3, T) for _ in range(N)]
    # rollouts3 = [sample_trajectory(G3, init3, init1, init2, T) for _ in range(N)]
    for i in range(N):
        np.save(os.path.join(path_static, f"mpc_traj1_{i}.npy"), rollouts1[i])
        np.save(os.path.join(path_static, f"mpc_traj2_{i}.npy"), rollouts2[i])
        np.save(os.path.join(path_static, f"mpc_traj3_{i}.npy"), rollouts3[i])
        # np.save(f"sampled_trajs/magail_nofinalpos_big/static_init/traj1_{i}.npy", rollouts1[i])
        # np.save(f"sampled_trajs/magail_nofinalpos_big/static_init/traj2_{i}.npy", rollouts2[i])
        # np.save(f"sampled_trajs/magail_nofinalpos_big/static_init/traj3_{i}.npy", rollouts3[i])

    # --- plot all rollouts --------------------------------
    # plt.figure(figsize=(6,6))
    # for r in rollouts1:
    #     plt.plot(r[:,0], r[:,1], color='blue', alpha=0.3)
    # for r in rollouts2:
    #     plt.plot(r[:,0], r[:,1], color='orange', alpha=0.3)
    # for r in rollouts3:
    #     plt.plot(r[:,0], r[:,1], color='green', alpha=0.3)
    # plt.title("Mode Collapse Check (fixed start→goal)")
    # plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid()
    # plt.show()
