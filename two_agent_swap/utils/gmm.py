from sklearn.mixture import GaussianMixture
import numpy as np
import csv
from joblib import dump, load

# Loading training trajectories
all_points = []
with open('data/trajs_noise1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x1, y1 = float(row[4]), float(row[5])
        x2, y2 = float(row[7]), float(row[8])
        all_points.append([x1, y1, x2, y2])

expert_data = np.array(all_points)

with open("data/mean_reactive.npy", "rb") as f:
    mean = np.load(f)
with open("data/std_reactive.npy", "rb") as f:
    std = np.load(f)
mean = np.concatenate([mean, mean])
std = np.concatenate([std, std])
expert_data = (expert_data - mean) / std

# Fit a GMM; choose the number of components (e.g., 3) based on your data.
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(expert_data)

# Save the GMM
dump(gmm, 'utils/expert_gmm.pkl')

# Function to compute probability density of a new state vector
def expert_likelihood(gmm_model, state_vector):
    # state_vector should be a 1D array of length 4.
    log_prob = gmm_model.score_samples(state_vector.reshape(1, -1))
    return np.exp(log_prob)[0]  # convert log-likelihood to probability

# # Example new vector (replace with your agent positions)
# new_state = np.array([0., 0.0, 20., 0.0])
# new_state_norm = (new_state - mean) / std
# # new_state_norm = np.array([-1.53335437, 0.02876356, 1.5113904, 0.08285503])
# print("Test vector:", new_state)
# print("Expert likelihood:", expert_likelihood(gmm, new_state_norm))

# new_state = np.array([7.5, 3.0, 12.5, 3.0])
# new_state_norm = (new_state - mean) / std
# print("Test vector:", new_state)
# print("Expert likelihood:", expert_likelihood(gmm, new_state_norm))

# new_state = np.array([10., 4.0, 10., 4.0])
# new_state_norm = (new_state - mean) / std
# print("Test vector:", new_state)
# print("Expert likelihood:", expert_likelihood(gmm, new_state_norm))

# new_state = np.array([5., 1.0, 15., -1.0])
# new_state_norm = (new_state - mean) / std
# print("Test vector:", new_state)
# print("Expert likelihood:", expert_likelihood(gmm, new_state_norm))
