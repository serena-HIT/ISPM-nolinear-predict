import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# # Define the Henon map function
# def henon_map(x, y, a, b):
#     """Returns the next state of the Henon map given current state and parameters."""
#     new_x = 1 - a * x**2 + y
#     new_y = b * x
#     return new_x, new_y

# # Henon map parameters
# a, b = 1.4, 0.3  # Standard Henon map parameters

# # Generate initial states randomly for 10000 samples
# initial_states = np.random.rand(50000, 2) * 2 - 1  # Random initial states in range [-1, 1]

# # Time length of 100 steps for each trajectory
# time_length = 500

# # Preallocate the dataset array
# X = np.empty((50000, time_length, 2))

# # Generate solutions and populate dataset
# for i, (x, y) in enumerate(initial_states):
#     trajectory = np.empty((time_length, 2))
#     trajectory[0] = [x, y]  # Set initial state

#     # Iterate through time steps
#     for t in range(1, time_length):
#         x, y = henon_map(x, y, a, b)
#         trajectory[t] = [x, y]

#     # Store the trajectory in the dataset
#     X[i] = trajectory
#     print("i = {}".format(i))

# # Report dataset shape to validate completion
# print(X.shape)  # or np.save("henon_map_data.npy", X) to save the dataset

# # plt.figure(figsize=(10, 5))
# # plt.plot(X[5, :, 0], X[5, :, 1], label="Henon Trajectory")
# # plt.title("Henon Map Trajectory for 100 Steps")
# # plt.xlabel("x")
# # plt.ylabel("y")
# # plt.legend()
# # plt.grid()
# # plt.show()
# # 保存数据集
# np.save("utility_LSPN/henon_map_data.npy", X)

X = np.load("utility_LSPN/henon_map_data.npy")
print(X.shape)
import numpy as np


def remove_invalid_trajectories(X):

    valid_trajectories = [trajectory for trajectory in X if not np.any(np.isinf(trajectory) | np.isnan(trajectory))]
    return np.array(valid_trajectories)

X_filtered = remove_invalid_trajectories(X)

np.save("henon_map_data_filtered.npy", X_filtered)

print(X.shape)
print(X_filtered[:20,:50,:])
np.save("utility_LSPN/henon_map_data_filtered.npy", X_filtered)
pass
