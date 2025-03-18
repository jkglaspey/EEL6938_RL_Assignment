import os
import matplotlib.pyplot as plt
import pandas as pd

def read_reward_data(file_path):
    """Reads the timestep and average reward from a given CSV file."""
    df = pd.read_csv(file_path)
    return df["timestep"], df["average_reward"]

def plot_rewards(file_paths, file_names, output_path):
    """Plots multiple average reward curves against timesteps."""
    plt.figure(figsize=(10, 5))
    
    for i in range(len(file_paths)):
        timesteps, avg_rewards = read_reward_data(file_paths[i])
        plt.plot(timesteps, avg_rewards, label=file_names[i], linewidth=1, alpha=0.9)
    
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.title("Training Convergence (Reward Function)")
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    #ax.set_ylim([-14, 0])
    plt.savefig(output_path)
    plt.show()
    print(f"Saved convergence plot to {output_path}")

reward_files = ["./logs_chunk_training/model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.005_gamma-0.99_ent-auto_arch-256-256_reward-0/training_log.csv",
                "./logs_chunk_training/model-0_chunk-100_lr-0.0003_batch-256_buffer-200000_tau-0.005_gamma-0.99_ent-auto_arch-256-256_reward-4/training_log.csv"
                ]
reward_filenames = ["SAC: Reward = Absolute Error", "SAC: Reward = Cubed Error"]
plot_rewards(reward_files, reward_filenames, "reward_convergence.png")
