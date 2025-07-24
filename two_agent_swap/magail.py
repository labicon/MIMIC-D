import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Generator MLP model for imitation learning
class Generator(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=4):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Example expert_data with updated format and 50 steps per trajectory
import csv
with open('data/full_traj_two_simple.csv', 'r') as file:
    reader = csv.reader(file)
    all_points = []
    for row in reader:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        all_points.append((x1, y1, x2, y2))

num_trajectories = 1000
points_per_trajectory = 100

expert_data = [
    all_points[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]

# Define goals for each agent
agent1_goal = [20, 0]
agent2_goal = [0, 0]

# Prepare the expert data with goal distance as input
X_expert = []
for trajectory in expert_data:
    for x1, y1, x2, y2 in trajectory:
        # Distances to goal
        dist_to_goal1 = np.sqrt((agent1_goal[0] - x1) ** 2 + (agent1_goal[1] - y1) ** 2)
        dist_to_goal2 = np.sqrt((agent2_goal[0] - x2) ** 2 + (agent2_goal[1] - y2) ** 2)
        X_expert.append([x1, y1, x2, y2])

X_expert = torch.tensor(X_expert, dtype=torch.float32)

# Initialize generator, discriminator, and optimizers
generator = Generator(input_size=6, hidden_size=64, output_size=4)
discriminator = Discriminator(input_size=4, hidden_size=64)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005)

# Loss function
bce_loss = nn.BCELoss()

# Training loop with batching and size alignment
num_epochs = 1000
batch_size = 128
num_batches = X_expert.size(0) // batch_size

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()

    for batch_idx in range(num_batches):
        # === Train Discriminator ===
        disc_optimizer.zero_grad()

        # Prepare real (expert) batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        real_batch = X_expert[start_idx:end_idx]
        real_labels = torch.ones(real_batch.size(0), 1)  # Match batch size

        # Discriminator on real data
        real_out = discriminator(real_batch)
        real_loss = bce_loss(real_out, real_labels)

        # Generate fake data batch
        fake_batch = []
        for x1, y1, x2, y2 in expert_data[batch_idx]:
            dist_to_goal1 = np.sqrt((agent1_goal[0] - x1) ** 2 + (agent1_goal[1] - y1) ** 2)
            dist_to_goal2 = np.sqrt((agent2_goal[0] - x2) ** 2 + (agent2_goal[1] - y2) ** 2)
            input_tensor = torch.tensor([x1, y1, x2, y2, dist_to_goal1, dist_to_goal2], dtype=torch.float32).unsqueeze(0)
            fake_batch.append(generator(input_tensor))

        fake_batch = torch.cat(fake_batch)
        fake_labels = torch.zeros(fake_batch.size(0), 1)  # Match fake batch size

        # Discriminator on fake data
        fake_out = discriminator(fake_batch.detach())
        fake_loss = bce_loss(fake_out, fake_labels)

        # Total discriminator loss
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # === Train Generator ===
        gen_optimizer.zero_grad()

        # Generator loss: trick discriminator into thinking fake is real
        gen_labels = torch.ones(fake_batch.size(0), 1)  # Label fake data as real
        gen_loss = bce_loss(discriminator(fake_batch), gen_labels)
        gen_loss.backward()
        gen_optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}")

# Generate trajectories
agent1_trajectory = []
agent2_trajectory = []
with torch.no_grad():
    x1, y1, x2, y2 = 0, 0, 20, 0  # Initial positions
    for _ in range(50):  # Generate 50 steps
        # Calculate distances to goals
        dist_to_goal1 = np.sqrt((agent1_goal[0] - x1) ** 2 + (agent1_goal[1] - y1) ** 2)
        dist_to_goal2 = np.sqrt((agent2_goal[0] - x2) ** 2 + (agent2_goal[1] - y2) ** 2)
        
        # Create input tensor
        input_tensor = torch.tensor([x1, y1, x2, y2, dist_to_goal1, dist_to_goal2], dtype=torch.float32).unsqueeze(0)
        
        # Predict next positions
        next_positions = generator(input_tensor).squeeze(0)  # Remove batch dimension
        
        # Unpack predictions
        x1, y1 = next_positions[:2].tolist()  # First two elements for Agent 1
        x2, y2 = next_positions[2:].tolist()  # Last two elements for Agent 2
        
        # Append to trajectories
        agent1_trajectory.append((x1, y1))
        agent2_trajectory.append((x2, y2))


# Convert trajectories to numpy arrays for plotting
agent1_trajectory = np.array(agent1_trajectory)
agent2_trajectory = np.array(agent2_trajectory)

# Plot the generated trajectories
plt.figure(figsize=(10, 6))
plt.plot(agent1_trajectory[:, 0], agent1_trajectory[:, 1], label='Agent 1 Trajectory')
plt.plot(agent2_trajectory[:, 0], agent2_trajectory[:, 1], label='Agent 2 Trajectory')

# Plot the start and end points
plt.scatter(0, 0, c='blue', s=100, label='Agent 1 Start')
plt.scatter(20, 0, c='cyan', s=100, label='Agent 1 Goal')
plt.scatter(20, 0, c='red', s=100, label='Agent 2 Start')
plt.scatter(0, 0, c='orange', s=100, label='Agent 2 Goal')

# Plot the obstacle
obstacle_circle = plt.Circle((10, 0), 4, color='gray', alpha=0.3)
plt.gca().add_patch(obstacle_circle)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Generated Trajectories for Agents with GAIL")
plt.grid(True)
plt.show()
