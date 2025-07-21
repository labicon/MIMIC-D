import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define the Generator (Policy) Network for Imitation Learning
class GeneratorNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(GeneratorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Discriminator Network
# Takes current state (2) + action (2) = 4 as input to judge the state-action pair
class DiscriminatorNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # Output a probability between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) # Sigmoid for probability output
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

# Prepare expert state-action pairs for the discriminator
# The discriminator takes (current_state, action) as input.
# The action is the difference between current_state and next_state if your expert data is states.
# In your original BC script, Y_train was the next state, so the action is Y_train - X_train[:, :2]
X_expert_discriminator1 = []
X_expert_discriminator2 = []

for i in range(len(expert_data_1)):
    for j in range(len(expert_data_1[i]) - 1):
        current_state = expert_data_1[i][j]
        next_state = expert_data_1[i][j + 1]
        action = next_state - current_state # Calculate action from state difference
        X_expert_discriminator1.append(np.hstack([current_state, action]))

for i in range(len(expert_data_2)):
    for j in range(len(expert_data_2[i]) - 1):
        current_state = expert_data_2[i][j]
        next_state = expert_data_2[i][j + 1]
        action = next_state - current_state # Calculate action from state difference
        X_expert_discriminator2.append(np.hstack([current_state, action]))

X_expert_discriminator1 = torch.tensor(np.array(X_expert_discriminator1), dtype=torch.float32).to(device)
X_expert_discriminator2 = torch.tensor(np.array(X_expert_discriminator2), dtype=torch.float32).to(device)

# Initialize Generators and Discriminators
generator1 = GeneratorNet(input_size=4, hidden_size=64, output_size=2)
discriminator1 = DiscriminatorNet(input_size=4, hidden_size=64, output_size=1) # Input: state (2) + action (2)
generator2 = GeneratorNet(input_size=4, hidden_size=64, output_size=2)
discriminator2 = DiscriminatorNet(input_size=4, hidden_size=64, output_size=1)

generator1, discriminator1 = generator1.to(device), discriminator1.to(device)
generator2, discriminator2 = generator2.to(device), discriminator2.to(device)

# Optimizers for Generator and Discriminator
# Different learning rates are common in GANs
gen_optimizer = optim.Adam(list(generator1.parameters()) + list(generator2.parameters()), lr=1e-5, weight_decay=1e-4)
disc_optimizer = optim.Adam(list(discriminator1.parameters()) + list(discriminator2.parameters()), lr=5e-5, weight_decay=1e-4)

# Loss Functions
# Binary Cross Entropy for discriminator
bce_loss = nn.BCELoss()

# Training Loop for MAGAIL
def train_magail(generator1, discriminator1, generator2, discriminator2,
                 gen_optimizer, disc_optimizer,
                 X_expert_discriminator1, X_expert_discriminator2,
                 initial_point1, final_point1, initial_point2, final_point2,
                 num_epochs=5000, trajectory_steps=100, noise_std=0.1):

    for epoch in range(num_epochs):
        # --- Train Discriminators ---
        disc_optimizer.zero_grad()

        # Expert data for Discriminator 1
        d1_expert_output = discriminator1(X_expert_discriminator1)
        d1_expert_loss = bce_loss(d1_expert_output, torch.ones_like(d1_expert_output))

        # Expert data for Discriminator 2
        d2_expert_output = discriminator2(X_expert_discriminator2)
        d2_expert_loss = bce_loss(d2_expert_output, torch.ones_like(d2_expert_output))

        # Generate fake trajectories for Discriminators
        # This part requires generating trajectories on the fly for the discriminator
        generated_trajectories1 = []
        generated_trajectories2 = []

        # It's crucial to detach the generated actions when training the discriminator
        # so that gradients don't flow back to the generator.
        with torch.no_grad():
            # Generate a single trajectory for each agent for discriminator training
            # You might want to generate a batch of trajectories for better training
            initial1_noisy = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
            final1_noisy = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
            initial2_noisy = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
            final2_noisy = final_point2 + noise_std * np.random.randn(*np.shape(final_point2))

            current_state1 = torch.tensor(np.hstack([initial1_noisy, final1_noisy]), dtype=torch.float32).unsqueeze(0).to(device)
            current_state2 = torch.tensor(np.hstack([initial2_noisy, final2_noisy]), dtype=torch.float32).unsqueeze(0).to(device)

            generated_actions_disc1 = []
            generated_actions_disc2 = []
            for _ in range(trajectory_steps - 1):
                # Predict next state, then compute action
                predicted_next_state1 = generator1(current_state1).squeeze(0)
                action1 = predicted_next_state1 - current_state1[:, :2].squeeze(0) # Action is (next_state - current_state)
                generated_actions_disc1.append(torch.hstack([current_state1[:, :2].squeeze(0), action1]))

                predicted_next_state2 = generator2(current_state2).squeeze(0)
                action2 = predicted_next_state2 - current_state2[:, :2].squeeze(0) # Action is (next_state - current_state)
                generated_actions_disc2.append(torch.hstack([current_state2[:, :2].squeeze(0), action2]))

                current_state1 = torch.tensor(np.hstack([predicted_next_state1.cpu().numpy(), final1_noisy]), dtype=torch.float32).unsqueeze(0).to(device)
                current_state2 = torch.tensor(np.hstack([predicted_next_state2.cpu().numpy(), final2_noisy]), dtype=torch.float32).unsqueeze(0).to(device)

            generated_data_disc1 = torch.stack(generated_actions_disc1)
            generated_data_disc2 = torch.stack(generated_actions_disc2)

        # Generated data for Discriminator 1
        d1_generated_output = discriminator1(generated_data_disc1)
        d1_generated_loss = bce_loss(d1_generated_output, torch.zeros_like(d1_generated_output))

        # Generated data for Discriminator 2
        d2_generated_output = discriminator2(generated_data_disc2)
        d2_generated_loss = bce_loss(d2_generated_output, torch.zeros_like(d2_generated_output))

        disc_loss = d1_expert_loss + d1_generated_loss + d2_expert_loss + d2_generated_loss
        disc_loss.backward()
        disc_optimizer.step()

        # --- Train Generators ---
        gen_optimizer.zero_grad()

        # Generate new trajectories for Generator training
        # This time, allow gradients to flow back to the generators
        initial1_noisy = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
        final1_noisy = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
        initial2_noisy = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
        final2_noisy = final_point2 + noise_std * np.random.randn(*np.shape(final_point2))

        current_state1 = torch.tensor(np.hstack([initial1_noisy, final1_noisy]), dtype=torch.float32).unsqueeze(0).to(device)
        current_state2 = torch.tensor(np.hstack([initial2_noisy, final2_noisy]), dtype=torch.float32).unsqueeze(0).to(device)

        gen_actions1 = []
        gen_actions2 = []
        for _ in range(trajectory_steps - 1):
            predicted_next_state1 = generator1(current_state1).squeeze(0)
            action1 = predicted_next_state1 - current_state1[:, :2].squeeze(0)
            gen_actions1.append(torch.hstack([current_state1[:, :2].squeeze(0), action1]))

            predicted_next_state2 = generator2(current_state2).squeeze(0)
            action2 = predicted_next_state2 - current_state2[:, :2].squeeze(0)
            gen_actions2.append(torch.hstack([current_state2[:, :2].squeeze(0), action2]))

            current_state1 = torch.tensor(np.hstack([predicted_next_state1.detach().cpu().numpy(), final1_noisy]), dtype=torch.float32).unsqueeze(0).to(device)
            current_state2 = torch.tensor(np.hstack([predicted_next_state2.detach().cpu().numpy(), final2_noisy]), dtype=torch.float32).unsqueeze(0).to(device)

        generated_data_gen1 = torch.stack(gen_actions1)
        generated_data_gen2 = torch.stack(gen_actions2)

        # Generator loss: try to maximize discriminator's output for fake data
        # Use log(D) for reward, so minimize -log(D) or maximize log(D)
        g_output1 = discriminator1(generated_data_gen1)
        g_loss1 = bce_loss(g_output1, torch.ones_like(g_output1)) # Try to make discriminator output 1 for fake data

        g_output2 = discriminator2(generated_data_gen2)
        g_loss2 = bce_loss(g_output2, torch.ones_like(g_output2)) # Try to make discriminator output 1 for fake data

        gen_loss = g_loss1 + g_loss2
        gen_loss.backward()
        gen_optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Disc Loss: {disc_loss.item():.4f}, Gen Loss: {gen_loss.item():.4f}')

    return generator1, discriminator1, generator2, discriminator2

print("Starting MAGAIL training...")
trained_generator1, trained_discriminator1, trained_generator2, trained_discriminator2 = train_magail(
    generator1, discriminator1, generator2, discriminator2,
    gen_optimizer, disc_optimizer,
    X_expert_discriminator1, X_expert_discriminator2,
    initial_point1, final_point1, initial_point2, final_point2,
    num_epochs=5000 # You might need more epochs for GANs
)
print("MAGAIL training complete.")

# Save the trained generator models
save_path_gen1 = "trained_models/magail/generator1_joint.pth"
save_path_gen2 = "trained_models/magail/generator2_joint.pth"
os.makedirs(os.path.dirname(save_path_gen1), exist_ok=True)
torch.save(trained_generator1.state_dict(), save_path_gen1)
torch.save(trained_generator2.state_dict(), save_path_gen2)
print("Trained generator models saved.")

# Load models (demonstration, already loaded above for testing purposes)
# model1 = GeneratorNet(input_size=4, hidden_size=64, output_size=2)
# model1.load_state_dict(torch.load(save_path_gen1, map_location='cpu'))
# model1.eval()

# model2 = GeneratorNet(input_size=4, hidden_size=64, output_size=2)
# model2.load_state_dict(torch.load(save_path_gen2, map_location='cpu'))
# model2.eval()


# Generate a New Trajectory Using the Trained Model
noise_std = 0.1
generated_trajectories1 = []
generated_trajectories2 = []

# Using the trained generator models
generator1.eval() # Set to evaluation mode
generator2.eval() # Set to evaluation mode

for _ in range(100):
    initial1 = initial_point1 + noise_std * np.random.randn(*np.shape(initial_point1))
    final1 = final_point1 + noise_std * np.random.randn(*np.shape(final_point1))
    initial2 = initial_point2 + noise_std * np.random.randn(*np.shape(initial_point2))
    final2 = final_point2 + noise_std * np.random.randn(*np.shape(final_point2)) # Typo fixed from original script

    with torch.no_grad():
        state1 = np.hstack([initial1, final1])  # Initial state + goal
        state1 = torch.tensor(state1, dtype=torch.float32).unsqueeze(0).to(device)
        traj1 = [initial1]

        state2 = np.hstack([initial2, final2])  # Initial state + goal
        state2 = torch.tensor(state2, dtype=torch.float32).unsqueeze(0).to(device)
        traj2 = [initial2]

        for _ in range(100 - 1):  # 100 steps total
            next_state1_pred = generator1(state1).cpu().numpy().squeeze()
            traj1.append(next_state1_pred)
            state1 = torch.tensor(np.hstack([next_state1_pred, final1]), dtype=torch.float32).unsqueeze(0).to(device)

            next_state2_pred = generator2(state2).cpu().numpy().squeeze()
            traj2.append(next_state2_pred)
            state2 = torch.tensor(np.hstack([next_state2_pred, final2]), dtype=torch.float32).unsqueeze(0).to(device)

    generated_trajectories1.append(np.array(traj1))
    generated_trajectories2.append(np.array(traj2))


# Plotting
plt.figure(figsize=(20, 8))
for i in range(len(generated_trajectories1)):
    traj1 = generated_trajectories1[i]
    traj2 = generated_trajectories2[i]
    plt.plot(traj1[:, 0], traj1[:, 1], 'b-', alpha=0.5)
    plt.plot(traj2[:, 0], traj2[:, 1], 'C1-', alpha=0.5)
    plt.scatter(traj1[0, 0], traj1[0, 1], c='green', s=10)  # Start point
    plt.scatter(traj1[-1, 0], traj1[-1, 1], c='red', s=10)  # End point
    plt.scatter(traj2[0, 0], traj2[0, 1], c='green', s=10)  # Start point
    plt.scatter(traj2[-1, 0], traj2[-1, 1], c='red', s=10)  # End point

ox, oy, r = obstacle
circle = plt.Circle((ox, oy), r, color='gray', alpha=0.3)
plt.gca().add_patch(circle)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()