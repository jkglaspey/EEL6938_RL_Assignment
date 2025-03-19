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
    plt.title("Training Convergence (Best Configurations)")
    plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.set_ylim([-10, 0])
    plt.savefig(output_path)
    plt.show()
    print(f"Saved convergence plot to {output_path}")

reward_files = ["./logs/model-0_chunk-200_lr-0.0003_batch-64_buffer-100000_tau-0.0001_gamma-0.9_ent-auto_arch-512-512_reward-4/training_log.csv",
                "./logs/model-1_chunk-200_lr-0.001_batch-256_buffer-200000_tau-0.005_gamma-0.9_ent-0.0_arch-256-256_reward-4/training_log.csv",
                "./logs/model-2_chunk-200_lr-0.0003_batch-128_buffer-1000000_tau-0.0001_gamma-0.99_ent-auto_arch-128-128_reward-4/training_log.csv",
                "./logs/model-3_chunk-200_lr-0.001_batch-64_buffer-50000_tau-0.0001_gamma-0.95_ent-auto_arch-64-64_reward-4/training_log.csv"
                ]
reward_filenames = ["SAC", "PPO", "TD3", "DDPG"]
plot_rewards(reward_files, reward_filenames, "best_convergence.png")
