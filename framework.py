import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.stats import truncnorm
import os
import wandb
import datetime
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import datetime




def transition_P_RKHS(state_space,action_space,P_kernel,alpha=0.5):
   
    grid_size = 10 # Grid size for fitting GP regression
    # Generate all possible input points in the grid with grid size 3
    values = np.linspace(0, 1, grid_size)
    X = np.array(list(product(values, repeat=3)))

    # Gaussian Process Regression (GPR)
    if P_kernel == "Matern_smoothness_1.5":
        kernel = Matern(length_scale=0.1, nu=1.5, length_scale_bounds="fixed")
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = Matern(length_scale=0.1, nu=2.5, length_scale_bounds="fixed")
    elif P_kernel =="RBF":
        kernel = RBF(length_scale=0.1,length_scale_bounds="fixed")

    gp = GaussianProcessRegressor(kernel=kernel)
    y = gp.sample_y(X, 1)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=alpha)
    gpr.fit(X, y)

    # Generate all possible input points in the grid 
    values = np.linspace(0, 1, len(state_space))
    all_possible_inputs = np.array(list(product(values, repeat=3)))

    # Predict for all possible input points
    all_predictions, _ = gpr.predict(all_possible_inputs, return_std=True)

    # Scale up and normalize the predictions
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    scaled_predictions = (all_predictions - min_prediction) / (max_prediction - min_prediction)

    num_state_action_pairs = len(all_possible_inputs) // len(state_space)
    for i in range(num_state_action_pairs):
        state_start_index = i * len(state_space)
        state_end_index = (i + 1) * len(state_space)
        state_predictions = scaled_predictions[state_start_index:state_end_index]
        scaled_predictions[state_start_index:state_end_index] = state_predictions / np.sum(state_predictions)

    P = scaled_predictions.reshape((len(state_space), len(action_space), len(state_space)))

    return P

# Define transition dynamics function which takes current state and action values as input and outputs the value of the next state
def transition_dynamics(current_state, action, transition_P,state_space,action_space): 

    # Get the index of the current state
    current_state_idx = np.argmin(np.abs(state_space - current_state))
    # Get the index of the action
    action_idx = np.argmin(np.abs(action_space - action))

    # Retrieve transition probabilities from the precomputed dynamics
    transition_probs = transition_P[current_state_idx, action_idx, :]

    # Sample next state based on transition probabilities
    next_state_idx = np.random.choice(np.arange(len(state_space)), p=transition_probs)

    next_state = state_space[next_state_idx]

    return next_state
   


def reward_RKHS(P_kernel,state_space,action_space,subdir=None,alpha=0.5):
    grid_size = 10 # Grid size for fitting GP regression

    # Generate all possible input points in the grid with grid size 3
    values = np.linspace(0, 1, grid_size)
    X = np.array(list(product(values, repeat=2)))


    # Gaussian Process Regression (GPR)
    if P_kernel == "Matern_smoothness_1.5":
        kernel = Matern(length_scale=0.1, nu=1.5, length_scale_bounds="fixed")
    elif P_kernel == "Matern_smoothness_2.5":
        kernel = Matern(length_scale=0.1, nu=2.5, length_scale_bounds="fixed")
    elif P_kernel =="RBF":
        kernel = RBF(length_scale=0.1,length_scale_bounds="fixed")

    gp = GaussianProcessRegressor(kernel=kernel)
    y = gp.sample_y(X, 1)

    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None,alpha=alpha)
    gpr.fit(X, y)

    # Generate all possible input points in the grid 
    values = np.linspace(0, 1, len(state_space))
    all_possible_inputs = np.array(list(product(values, repeat=2)))

    # Predict for all possible input points
    all_predictions, _ = gpr.predict(all_possible_inputs, return_std=True)
    y_pred, _=gpr.predict(X,return_std=True)
    mse = mean_squared_error(y,y_pred)

    # Scale up and normalize the predictions
    min_prediction = np.min(all_predictions)
    max_prediction = np.max(all_predictions)
    scaled_predictions = (all_predictions - min_prediction) / (max_prediction - min_prediction)
    r = scaled_predictions.reshape((len(state_space), len(action_space)))
    #plot_reward_gp_3d(r, state_space, action_space,mse, subdir,X,y)

    return r



# Value iteration algorithm
def value_iteration_episodic(state_space, action_space, r, P, H=10):
    V = np.zeros_like(state_space)  # Initialize value function
    for h in range(H):  # Iterate over time steps
        V_new = np.zeros_like(V)  # Initialize new value function
        for s_idx, s in enumerate(state_space):
            Q_values = []
            for a_idx, a in enumerate(action_space):
                expected_return = 0
                for next_s_idx, next_s in enumerate(state_space):
                    reward = r[s_idx,a_idx]
                    expected_return += P[s_idx, a_idx, next_s_idx] * (reward + V[next_s_idx])
                Q_values.append(expected_return)
            V_new[s_idx] = max(Q_values)  # Update value function for state s
        V = V_new  # Update value function after each iteration

    return V # V is indexed by state index


def plot_reward_gp_3d(r, state_space, action_space, save_dir=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    action_mesh, state_mesh = np.meshgrid(state_space, action_space)
    surf = ax.plot_surface(state_mesh, action_mesh, r, cmap='viridis',vmin=0,vmax=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #     # Scatter plot of predicted rewards
    # state_indices = np.arange(len(state_space))
    # action_indices = np.arange(len(action_space))
    # for state_idx in state_indices:
    #     for action_idx in action_indices:
    #         x = state_space[state_idx]
    #         y = action_space[action_idx]
    #         z = r[state_idx, action_idx]
    #         ax.scatter(x, y, z, color='red', s=50)
    
    ax.set_xlabel('s',fontsize=20)
    ax.set_ylabel('a',fontsize=20)
    ax.set_zlabel('r(s,a)',fontsize=20)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(save_dir, f"reward_gp_3d_{timestamp}.png")
        plt.savefig(save_path)
        plt.close()
    else:
        wandb.log({"Reward Surface 3D": wandb.Image(plt)})
    
   

def plot_transition_probabilities(P, state_space,action_space, save_dir=None):

    # Choose state and action indices
    state_indices = [0, 50, 99]  # Indices of states in state_space
    action_indices = [0, 50, 99]  # Indices of actions in action_space


# Plot transition probabilities for each (s, a) pair
    for state_idx in state_indices:
        for action_idx in action_indices:
    # Get the transition probabilities for the specified state-action pair
            transition_probs = P[state_idx, action_idx, :]

            state_labels = [str(round(float(state[0]), 4)) for state in state_space]
            action_labels = [str(round(float(action[0]), 4)) for action in action_space]
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.bar(state_labels, transition_probs)
            plt.xlabel("s'",fontsize=20)
            # plt.rcParams['text.usetex'] = True
            # plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
            
            plt.ylabel(r"$P(s' \mid s={}, a={})$".format(state_labels[state_idx], action_labels[action_idx]), fontsize=20)
            
            plt.xticks([state_labels[0], state_labels[-1]])
            
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"P_s{state_idx}_a{action_idx}.png")
                plt.savefig(save_path)  # Save the figure with a unique name
                plt.close()
            else: 
                # Convert the Matplotlib figure to a wandb Image and log it
                wandb.log({f"Transition Probabilities for State-Action Pair (s={state_space[state_idx]}, a={action_space[action_idx]})": wandb.Image(plt)})

