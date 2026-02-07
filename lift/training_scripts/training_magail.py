# ==== Multi-Agent GAIL (sequence-aware) ====
# Two-arm setting; generator/discriminator handle horizon H sequences.

import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Repro & device
# -------------------------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = True

# -------------------------
# Helpers
# -------------------------
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

# -------------------------
# Hyperparameters
# -------------------------
hidden_size  = 256
lr_G         = 1e-4
lr_D         = 1e-4
target_steps = 50_000
batch_size   = 32
H            = 25   # horizon
T            = 700  # total time steps (for tiling attrs)
disc_layers  = 6    # discriminator temporal depth
gen_layers   = 8    # generator temporal depth
dropout      = 0.1
expansion    = 4

# -------------------------
# Load & prep expert data
# -------------------------
expert_data = np.load("data/expert_actions_newslower_20.npy")  # (N, full_T, 14)
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)  # (N*, H, 7)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)  # (N*, H, 7)

# Normalize per-arm
mean_arm1 = np.mean(expert_data1, axis=(0,1)); std_arm1 = np.std(expert_data1, axis=(0,1)) + 1e-8
mean_arm2 = np.mean(expert_data2, axis=(0,1)); std_arm2 = np.std(expert_data2, axis=(0,1)) + 1e-8
expert_data1 = (expert_data1 - mean_arm1) / std_arm1
expert_data2 = (expert_data2 - mean_arm2) / std_arm2

# Env-ish stub (kept for clarity)
class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"
env = TwoArmLift()

# Tensors
actions1 = torch.tensor(expert_data1, dtype=torch.float32)  # (N, H, 7)
actions2 = torch.tensor(expert_data2, dtype=torch.float32)  # (N, H, 7)

with open("data/pot_start_newslower_20.npy", "rb") as f:
    pot = np.load(f)  # (N_base, pot_dim)

pot_mean = np.mean(pot, axis=0); pot_std = np.std(pot, axis=0) + 1e-8
pot = (pot - pot_mean) / pot_std
pot_dim = pot.shape[1]

# Tile pot to match number of sub-trajectories (we created n_subtraj per original traj)
# We assume expert_data1/expert_data2 were built by sliding over each original traj.
# If original had N0 trajectories and horizon L, then N = N0 * L. We repeat pot rows L times.
N = actions1.shape[0]
N0 = pot.shape[0]
assert N % N0 == 0, "Tiling mismatch: ensure pot matches base trajectories."
repeat_factor = N // N0
obs = np.repeat(pot, repeats=repeat_factor, axis=0)  # (N, pot_dim)

# Build attributes like BC: [init1, init2, pot]
obs_init1 = expert_data1[:, 0, :]  # (N, 7)
obs_init2 = expert_data2[:, 0, :]  # (N, 7)
obs1 = np.hstack([obs_init1, obs_init2, obs])  # (N, 7+7+pot_dim)
obs2 = np.hstack([obs_init2, obs_init1, obs])

attr1 = torch.tensor(obs1, dtype=torch.float32)
attr2 = torch.tensor(obs2, dtype=torch.float32)

input_size = attr1.shape[1]   # s_dim for each agent's conditioning input
action_dim = actions1.shape[-1]

# -------------------------
# Sequence blocks
# -------------------------
class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * expansion)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h); h = self.act(h); h = self.drop(h)
        h = self.fc2(h); h = self.drop(h)
        return x + h

# -------------------------
# Generator: attr -> (B, H, action_dim)
# -------------------------
class SeqGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=256, action_dim=7, horizon=25,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon
        self.inp  = nn.Linear(input_size, hidden_size)
        self.step_embed = nn.Embedding(horizon, hidden_size)
        self.blocks = nn.ModuleList([MLPBlock(hidden_size, dropout, expansion) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, attrs):  # attrs: (B, input_size)
        B = attrs.size(0); dev = attrs.device
        h = self.inp(attrs)                   # (B, hidden)
        h = h.unsqueeze(1).expand(B, self.horizon, -1)  # (B, H, hidden)
        steps = torch.arange(self.horizon, device=dev)  # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)     # (B, H, hidden)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)                    # (B, H, action_dim)
        return out

# -------------------------
# Discriminator: (attr, actions_seq) -> per-step logits (B, H)
# -------------------------
class SeqDiscriminator(nn.Module):
    def __init__(self, input_size, action_dim=7, hidden_size=256, horizon=25,
                 num_layers=6, dropout=0.1, expansion=4, pool='mean'):
        super().__init__()
        self.horizon = horizon
        self.pool = pool
        # Encode attributes and broadcast over time
        self.attr_in  = nn.Linear(input_size, hidden_size)
        self.step_embed = nn.Embedding(horizon, hidden_size)
        # Fuse per-step action
        self.fuse = nn.Linear(hidden_size + action_dim, hidden_size)
        # Temporal processing (per-step MLP blocks)
        self.blocks = nn.ModuleList([MLPBlock(hidden_size, dropout, expansion) for _ in range(num_layers)])
        # Per-step logits
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, attrs, actions):  # attrs: (B, input_size), actions: (B, H, action_dim)
        B, H, A = actions.shape
        assert H == self.horizon, f"H mismatch: got {H}, expected {self.horizon}"
        dev = actions.device
        h = self.attr_in(attrs)                         # (B, hidden)
        h = h.unsqueeze(1).expand(B, H, -1)             # (B, H, hidden)
        steps = torch.arange(H, device=dev)
        h = h + self.step_embed(steps).unsqueeze(0)     # (B, H, hidden)
        h = torch.cat([h, actions], dim=-1)             # (B, H, hidden + A)
        h = self.fuse(h)                                 # (B, H, hidden)
        for blk in self.blocks:
            h = blk(h)                                  # (B, H, hidden)
        logits_step = self.head(h).squeeze(-1)          # (B, H)
        if self.pool == 'mean':
            logits = logits_step.mean(dim=1)            # (B,)
        elif self.pool == 'sum':
            logits = logits_step.sum(dim=1)
        else:
            logits = logits_step[:, -1]                 # last step
        return logits, logits_step

# -------------------------
# Dataloaders (per agent)
# -------------------------
dataset1 = TensorDataset(attr1, actions1)
dataset2 = TensorDataset(attr2, actions2)

loader1  = DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=max(1, os.cpu_count()//2), pin_memory=(device.type=="cuda"),
                      persistent_workers=True)
loader2  = DataLoader(dataset2, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=max(1, os.cpu_count()//2), pin_memory=(device.type=="cuda"),
                      persistent_workers=True)

# -------------------------
# Build models
# -------------------------
G1 = SeqGenerator(input_size, hidden_size, action_dim, H, gen_layers, dropout, expansion).to(device)
G2 = SeqGenerator(input_size, hidden_size, action_dim, H, gen_layers, dropout, expansion).to(device)
D  = SeqDiscriminator(input_size, action_dim, hidden_size, H, disc_layers, dropout, expansion, pool='mean').to(device)

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=lr_G, betas=(0.5, 0.999), weight_decay=0.0)
optD = optim.Adam(D.parameters(),                                   lr=lr_D, betas=(0.5, 0.999), weight_decay=0.0)
bce_logits = nn.BCEWithLogitsLoss()

# -------------------------
# Training
# -------------------------
steps_per_epoch = max(len(loader1), len(loader2))
n_epochs = max(1, math.ceil(target_steps / max(1, steps_per_epoch)))

print(steps_per_epoch, n_epochs)

for epoch in range(1, n_epochs+1):
    G1.train(); G2.train(); D.train()
    it1 = iter(loader1); it2 = iter(loader2)
    lossD_sum = 0.0; lossG_sum = 0.0

    for i in range(steps_per_epoch):
        try: a1_attrs, a1_true = next(it1)
        except StopIteration: it1 = iter(loader1); a1_attrs, a1_true = next(it1)
        try: a2_attrs, a2_true = next(it2)
        except StopIteration: it2 = iter(loader2); a2_attrs, a2_true = next(it2)

        a1_attrs = a1_attrs.to(device, non_blocking=True)  # (B, input_size)
        a2_attrs = a2_attrs.to(device, non_blocking=True)
        a1_true  = a1_true.to(device, non_blocking=True)   # (B, H, A)
        a2_true  = a2_true.to(device, non_blocking=True)

        if i%5 == 0:
            # ---- Train D ----
            with torch.no_grad():
                a1_fake = G1(a1_attrs)  # (B, H, A)
                a2_fake = G2(a2_attrs)

            r1, _ = D(a1_attrs, a1_true)    # (B,)
            f1, _ = D(a1_attrs, a1_fake)
            r2, _ = D(a2_attrs, a2_true)
            f2, _ = D(a2_attrs, a2_fake)

            # import ipdb; ipdb.set_trace()  # IGNORE

            ones1 = torch.ones_like(r1); zeros1 = torch.zeros_like(f1)
            ones2 = torch.ones_like(r2); zeros2 = torch.zeros_like(f2)

            lossD1 = 0.5 * (bce_logits(r1, ones1) + bce_logits(f1, zeros1))
            lossD2 = 0.5 * (bce_logits(r2, ones2) + bce_logits(f2, zeros2))
            lossD  = lossD1 + lossD2

            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

        # ---- Train G ----
        a1_fake = G1(a1_attrs)
        a2_fake = G2(a2_attrs)
        f1, _ = D(a1_attrs, a1_fake)
        f2, _ = D(a2_attrs, a2_fake)

        # Generator tries to make D output "real"
        lossG = bce_logits(f1, torch.ones_like(f1)) + bce_logits(f2, torch.ones_like(f2))

        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()

        lossD_sum += lossD.item(); lossG_sum += lossG.item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"[{epoch:04d}/{n_epochs}] lossD={lossD_sum/steps_per_epoch:.4f}  lossG={lossG_sum/steps_per_epoch:.4f}")

# -------------------------
# Save
# -------------------------
out_dir = "trained_models/magail_seq"
os.makedirs(out_dir, exist_ok=True)
torch.save(G1.state_dict(), os.path.join(out_dir, "G1.pth"))
torch.save(G2.state_dict(), os.path.join(out_dir, "G2.pth"))
torch.save(D.state_dict(),  os.path.join(out_dir, "D.pth"))
print(f"Saved to {out_dir}")

# -------------------------
# Notes:
# - The discriminator pools per-step logits by mean; switch to 'sum' or 'last' via pool=...
# - You can add gradient penalty, spectral norm, or use WGAN loss if training is unstable.
# - If you later have true per-step states, replace attrs broadcast with those states per step.
# - Keep your BC normalizers (mean_arm*, std_arm*) to denorm outputs at rollout time.
# -------------------------
