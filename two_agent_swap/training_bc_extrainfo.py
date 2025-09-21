import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Define the Neural Network for Imitation Learning
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
        x = self.fc3(x)
        return x

# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Expert demonstration loading
expert_data_1 = np.load('data/expert_data1_100_traj.npy')
expert_data_2 = np.load('data/expert_data2_100_traj.npy')
X_train1 = []
Y_train1 = []
X_train2 = []
Y_train2 = []

for i in range(len(expert_data_1)):
    for j in range(len(expert_data_1[i]) - 1):
        X_train1.append(np.hstack([expert_data_1[i][j], expert_data_1[i][-1], expert_data_2[i][j]]))  # Current state + goal + other agent's current position
        Y_train1.append(expert_data_1[i][j + 1])  # Next state
X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32)  # Shape: (N, 2)

for i in range(len(expert_data_2)):
    for j in range(len(expert_data_2[i]) - 1):
        X_train2.append(np.hstack([expert_data_2[i][j], expert_data_2[i][-1], expert_data_1[i][j]]))  # Current state + goal + other agent's current position
        Y_train2.append(expert_data_2[i][j + 1])  # Next state
X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
model1 = ImitationNet(input_size=6, hidden_size=64, output_size=2)
model2 = ImitationNet(input_size=6, hidden_size=64, output_size=2)
criterion = nn.MSELoss()  # Mean Squared Error Loss
all_params = []
all_params += list(model1.parameters())
all_params += list(model2.parameters())
optimizer = optim.Adam(all_params, lr=0.001, weight_decay=1e-4)

model1, model2 = model1.to(device), model2.to(device)
X_train1, Y_train1 = X_train1.to(device), Y_train1.to(device)
X_train2, Y_train2 = X_train2.to(device), Y_train2.to(device)

# Train the Model
def train_model(model, optimizer, criterion, X_train, Y_train, num_epochs=5000):
    losses = []

    for epoch in range(num_epochs):
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model, losses

def joint_train(model1, model2, optimizer, criterion, X_train1, Y_train1, X_train2, Y_train2, num_epochs=10000):
    losses1 = []
    losses2 = []

    for epoch in range(num_epochs):
        loss_total = 0.0

        predictions1 = model1(X_train1)
        loss1 = criterion(predictions1, Y_train1)
        loss_total += loss1

        predictions2 = model2(X_train2)
        loss2 = criterion(predictions2, Y_train2)
        loss_total += loss2

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        losses1.append(loss1.item())
        losses2.append(loss2.item())

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')

    return model1, model2, losses1, losses2

# trained_model1, losses1 = train_model(model1, optimizer1, criterion, X_train1, Y_train1)
# trained_model2, losses2 = train_model(model2, optimizer2, criterion, X_train2, Y_train2)
# trained_model1, trained_model2, losses1, losses2 = joint_train(model1, model2, optimizer, criterion, X_train1, Y_train1, X_train2, Y_train2)


save_path1 = "trained_models/bc/bc_extrainfo1.pth"
save_path2 = "trained_models/bc/bc_extrainfo2.pth"
# torch.save(trained_model1.state_dict(), save_path1)
# torch.save(trained_model2.state_dict(), save_path2)

model1 = ImitationNet(input_size=6, hidden_size=64, output_size=2)
model1.load_state_dict(torch.load(save_path1, map_location='cpu'))
model1.eval()

model2 = ImitationNet(input_size=6, hidden_size=64, output_size=2)
model2.load_state_dict(torch.load(save_path2, map_location='cpu'))
model2.eval()

# Generate a New Trajectory Using the Trained Model
for s in range(10):
    seed = s * 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    path = f"sampled_trajs/bc_extrainfo_seed{seed}"
    os.makedirs(path, exist_ok=True)

    noise_std = 0.4
    generated_trajectories1 = []
    generated_trajectories2 = []

    for i in range(100):
        initial1 = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
        final1 = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
        initial2 = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
        final2 = final_point2 + noise_std * np.random.randn(*np.shape(final_point2))
        with torch.no_grad():
            state1 = np.hstack([initial1, final1, initial2])  # Initial state + goal
            state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0)
            traj1 = [initial1]

            state2 = np.hstack([initial2, final2, initial1])  # Initial state + goal
            state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0)
            traj2 = [initial2]

            for _ in range(100 - 1):  # 100 steps total
                next_state1 = model1(state1).numpy().squeeze()
                traj1.append(next_state1)

                next_state2 = model2(state2).numpy().squeeze()
                traj2.append(next_state2)

                state1 = torch.tensor(np.hstack([next_state1, final1, next_state2]), dtype=torch.float32).unsqueeze(0)
                state2 = torch.tensor(np.hstack([next_state2, final2, next_state1]), dtype=torch.float32).unsqueeze(0)

        generated_trajectories1.append(np.array(traj1))
        np.save(os.path.join(path, f"mpc_traj1_{i}.npy"), np.array(traj1))
        generated_trajectories2.append(np.array(traj2))
        np.save(os.path.join(path, f"mpc_traj2_{i}.npy"), np.array(traj2))


# # Plotting
# plt.figure(figsize=(20, 8))
# for i in range(len(generated_trajectories1)):
#     traj1 = generated_trajectories1[i]
#     traj2 = generated_trajectories2[i]
#     plt.plot(traj1[:, 0], traj1[:, 1], 'b-', alpha=0.5)
#     plt.plot(traj2[:, 0], traj2[:, 1], 'C1-', alpha=0.5)
#     plt.scatter(traj1[0, 0], traj1[0, 1], c='green', s=10)  # Start point
#     plt.scatter(traj1[-1, 0], traj1[-1, 1], c='red', s=10)  # End point
#     plt.scatter(traj2[0, 0], traj2[0, 1], c='green', s=10)  # Start point
#     plt.scatter(traj2[-1, 0], traj2[-1, 1], c='red', s=10)  # End point

# ox, oy, r = obstacle
# circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
# plt.gca().add_patch(circle)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()