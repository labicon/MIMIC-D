import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Behavioral Cloning Network
class ImitationNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(ImitationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Load expert demonstrations
expert_data1 = np.load('data/expert_data1_400_traj_06_noise.npy')  # (n_traj, horizon, 2)
expert_data2 = np.load('data/expert_data2_400_traj_06_noise.npy')
expert_data3 = np.load('data/expert_data3_400_traj_06_noise.npy')

# Build BC training data: (current_state, final_state, other_agent_states) -> next_state
def build_bc_data(expert_data, expert_data2, expert_data3):
    X, Y = [], []
    for i in range(len(expert_data)):
        traj = expert_data[i]
        traj2 = expert_data2[i]
        traj3 = expert_data3[i]
        final = traj[-1]
        for t in range(len(traj)-1):
            X.append(np.hstack([traj[t], final, traj2[t], traj3[t]]))
            Y.append(traj[t+1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X1, Y1 = build_bc_data(expert_data1, expert_data2, expert_data3)
X2, Y2 = build_bc_data(expert_data2, expert_data1, expert_data3)
X3, Y3 = build_bc_data(expert_data3, expert_data1, expert_data2)

# Convert to torch tensors
X1, Y1 = torch.from_numpy(X1).to(device), torch.from_numpy(Y1).to(device)
X2, Y2 = torch.from_numpy(X2).to(device), torch.from_numpy(Y2).to(device)
X3, Y3 = torch.from_numpy(X3).to(device), torch.from_numpy(Y3).to(device)

# Initialize models and optimizer
input_size = X1.shape[1]  # should be 8
model1 = ImitationNet(input_size, 2000, 2).to(device)
model2 = ImitationNet(input_size, 2000, 2).to(device)
model3 = ImitationNet(input_size, 2000, 2).to(device)
params = list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# Joint training for three agents
def joint_train(models, optimizer, criterion, datasets, num_epochs=5000):
    Xs, Ys = zip(*datasets)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0
        for model, X, Y in zip(models, Xs, Ys):
            pred = model(X)
            loss += criterion(pred, Y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return models

model1, model2, model3 = joint_train(
    [model1, model2, model3], optimizer, criterion,
    [(X1, Y1), (X2, Y2), (X3, Y3)], num_epochs=5000
)

# Save trained models
save_path1 = "trained_models/bc/bc_extrainfo1.pth"
save_path2 = "trained_models/bc/bc_extrainfo2.pth"
save_path3 = "trained_models/bc/bc_extrainfo3.pth"

torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)
torch.save(model3.state_dict(), save_path3)

model1 = ImitationNet(input_size=8, hidden_size=2000, output_size=2).to(device)
model1.load_state_dict(torch.load(save_path1, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model1.eval()

model2 = ImitationNet(input_size=8, hidden_size=2000, output_size=2).to(device)
model2.load_state_dict(torch.load(save_path2, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model2.eval()

model3 = ImitationNet(input_size=8, hidden_size=2000, output_size=2).to(device)
model3.load_state_dict(torch.load(save_path3, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
model3.eval()

# Sampling parameters
initial_point_1 = np.array([0.0, 2.0])
final_point_1 = np.array([2.0, 0.0])
initial_point_2 = np.array([0.75, -2.0])
final_point_2 = np.array([0.75, 2.0])
initial_point_3 = np.array([-0.25, 0.75])
final_point_3 = np.array([1.75, 0.75])

noise_std = 0.6
n_samples = 100
traj_length = 100
threshold = 0.75

for s in range(10):
    seed = s * 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    path = f"sampled_trajs/bc_extrainfo_seed{seed}"
    os.makedirs(path, exist_ok=True)

    generated = {1: [], 2: [], 3: []}
    for i in range(n_samples):
        while True:
            initial1 = initial_point_1 + np.random.uniform(-noise_std, noise_std, size=(2,))    
            initial2 = initial_point_2 + np.random.uniform(-noise_std, noise_std, size=(2,))
            initial3 = initial_point_3 + np.random.uniform(-noise_std, noise_std, size=(2,))
            final1 = final_point_1 + np.random.uniform(-noise_std, noise_std, size=(2,))    
            final2 = final_point_2 + np.random.uniform(-noise_std, noise_std, size=(2,))
            final3 = final_point_3 + np.random.uniform(-noise_std, noise_std, size=(2,))

            d_init12 = np.linalg.norm(initial1 - initial2)
            d_init13 = np.linalg.norm(initial1 - initial3)
            d_init23 = np.linalg.norm(initial2 - initial3)
            d_final12 = np.linalg.norm(final1 - final2)
            d_final13 = np.linalg.norm(final1 - final3)
            d_final23 = np.linalg.norm(final2 - final3)

            if (d_init12 > threshold and d_init13 > threshold and d_init23 > threshold and
                d_final12 > threshold and d_final13 > threshold and d_final23 > threshold):
                break

        traj1 = [initial1]
        traj2 = [initial2]
        traj3 = [initial3]
        s1 = torch.from_numpy(np.hstack([initial1, final1, initial2, initial3])).float().to(device).unsqueeze(0)
        s2 = torch.from_numpy(np.hstack([initial2, final2, initial1, initial3])).float().to(device).unsqueeze(0)
        s3 = torch.from_numpy(np.hstack([initial3, final3, initial1, initial2])).float().to(device).unsqueeze(0)

        for _ in range(traj_length-1):
            with torch.no_grad():
                next1 = model1(s1).cpu().numpy().squeeze()
                next2 = model2(s2).cpu().numpy().squeeze()
                next3 = model3(s3).cpu().numpy().squeeze()
            traj1.append(next1)
            traj2.append(next2)
            traj3.append(next3)
            s1 = torch.from_numpy(np.hstack([next1, final1, next2, next3])).float().to(device).unsqueeze(0)
            s2 = torch.from_numpy(np.hstack([next2, final2, next1, next3])).float().to(device).unsqueeze(0)
            s3 = torch.from_numpy(np.hstack([next3, final3, next1, next2])).float().to(device).unsqueeze(0)

        generated[1].append(np.array(traj1))
        np.save(os.path.join(path, f"mpc_traj1_{i}.npy"), np.array(traj1))
        # np.save(f"sampled_trajs/bc_nofinalpos_big/traj1_{i}.npy", np.array(traj1))
        generated[2].append(np.array(traj2))
        np.save(os.path.join(path, f"mpc_traj2_{i}.npy"), np.array(traj2))
        # np.save(f"sampled_trajs/bc_nofinalpos_big/traj2_{i}.npy", np.array(traj2))
        generated[3].append(np.array(traj3))
        np.save(os.path.join(path, f"mpc_traj3_{i}.npy"), np.array(traj3))

        # np.save(f"sampled_trajs/bc_nofinalpos_big/traj3_{i}.npy", np.array(traj3))

# # Plot all samples together
# plt.figure(figsize=(8, 6))
# for i in range(n_samples):
#     plt.plot(generated[1][i][:, 0], generated[1][i][:, 1], 'b-', alpha=0.3)
#     plt.plot(generated[2][i][:, 0], generated[2][i][:, 1], 'g-', alpha=0.3)
#     plt.plot(generated[3][i][:, 0], generated[3][i][:, 1], 'r-', alpha=0.3)
# # plot start/end markers
# for c, (init, fin) in zip(['b','g','r'], [(initial_point1,final_point1),(initial_point2,final_point2),(initial_point3,final_point3)]):
#     plt.scatter(init[0], init[1], c=c)
#     plt.scatter(fin[0], fin[1], c=c, marker='x')
# plt.xlabel('X'); plt.ylabel('Y'); plt.title('BC Sampled Trajectories')
# plt.grid(True)
# plt.show()
