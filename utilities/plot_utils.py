import matplotlib.pyplot as plt
import torch
import pickle
import os

def smooth(input, weight):
    """
    exponential moving average for smoothing
    https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    """
    last = input[0]
    smoothed = []
    for point in input:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                   
        last = smoothed_val
        
    return smoothed

def plot_episode_rewards(rewards, title, smoothing_weight=0.9):
    """
    Plot the episode rewards over episodes with a smoothed version.

    Inputs:
        rewards (list): List of episode rewards.
        title (str): Title of the plot.
        window_size (int): Size of the smoothing window (default is 10).
    """
    # Convert rewards list to a PyTorch tensor for processing
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot episode rewards
    ax.plot(rewards_tensor.numpy(), label='Episode Rewards', alpha=0.5)

    # Smooth rewards using a moving average (low-pass filter)
    smoothed_rewards = smooth(rewards_tensor, smoothing_weight)

    # Plot smoothed rewards
    ax.plot(smoothed_rewards, label='Smoothed Rewards', color='red')

    # Add labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)

    ax.legend()

    # Show plot
    plt.show()


def plot_actor_critic_losses(actor_losses, critic_losses):
    """
    Plot the loss for actor network and critic network on separate subplots.

    Inputs:
        actor_losses (list): List of actor network losses.
        critic_losses (list): List of critic network losses.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

    # Plot actor network loss on the first subplot
    ax1.plot(actor_losses, label='Actor Loss', color='b')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Actor Network Loss')

    # Plot critic network loss on the second subplot
    ax2.plot(critic_losses, label='Critic Loss', color='r')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Critic Network Loss')

    # Adjust layout and display the legend
    fig.tight_layout()
    plt.show()

def plot_learning_rates(learning_rates_actor, learning_rates_critic):
    # Create x-axis (epochs or iterations)
    epochs = list(range(1, len(learning_rates_actor) + 1))  # Assuming epochs as x-axis
    
    # Plot learning rates for actor and critic
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates_actor, label='Actor Learning Rates', color='b')
    plt.plot(epochs, learning_rates_critic, label='Critic Learning Rates', color='r')
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Actor and Critic Learning Rates over Epochs')
    
    plt.legend()

    plt.grid(True)
    plt.show()

def plot_actions_histogram(actions, n_actions, action_names=None):
    # Create histogram
    plt.hist(actions, bins=n_actions, color='blue', edgecolor='black', alpha=0.7)
    
    # Set x-axis labels
    if action_names is not None:
        plt.xticks(range(n_actions), [action_names[i] for i in range(n_actions)])
    else:
        plt.xticks(range(n_actions))
    
    # Add title and labels
    plt.title('Histogram of Actions Taken')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    
    # Show plot
    plt.show()

def plot_entropy_over_steps(entropy_list):
    """
    Plot entropy values over the number of train steps.
    
    Parameters:
        entropy_list (list): List of entropy values.
    """
    steps = list(range(1, len(entropy_list) + 1))
    plt.plot(steps, entropy_list, marker='o', linestyle='-')
    plt.xlabel('Number of Train Steps')
    plt.ylabel('Entropy')
    plt.title('Entropy over Train Steps')
    plt.grid(True)
    plt.show()

def save_plot_data(environment_name, agent_name, data_name, data):
        path = os.getcwd() + f'/{environment_name}/{agent_name}/'
        file_name = path + f'{environment_name}_{agent_name}_{data_name}.pkl'
        os.makedirs(path, exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

        print(f"{agent_name} Results saved to '{file_name}'")

def load_plot_data(environment_name, agent_name, data_name):
    path = os.getcwd() + f'/{environment_name}/{agent_name}/'
    file_name = path + f'{environment_name}_{agent_name}_{data_name}.pkl'
    os.makedirs(path, exist_ok=True)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_pickle_files(pickle_file_path_list):
    for pickle_file_path in pickle_file_path_list:
        reward_data = load_plot_data(pickle_file_path[0], pickle_file_path[1], pickle_file_path[2])
        plt.plot(reward_data)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('All Reward Plots')
    plt.grid(True)
    plt.show()
