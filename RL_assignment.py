import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO, TD3, DDPG
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"
ERROR_SELECTION = 0

# Force-generate a 1200-step sinusoidal + noise speed profile
if not os.path.exists(CSV_FILE):
    speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
    df_fake = pd.DataFrame({"speed": speeds})
    df_fake.to_csv(CSV_FILE, index=False)
    print(f"Created {CSV_FILE} with {DATA_LEN} steps.")
else:
    print(f"{CSV_FILE} already exists. Skipping creation.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reset: Dependent on reward function called
    """

    def __init__(self, episodes_list, delta_t=1.0):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        self.ref_speed = self.current_episode[self.step_idx]

        # 1: Absolute error
        if ERROR_SELECTION == 0:
            error = abs(self.current_speed - self.ref_speed)
            reward = -error

        # 2: Squared error
        elif ERROR_SELECTION == 1:
            error = self.current_speed - self.ref_speed
            reward = -error**2

        # 3: Exponential error
        elif ERROR_SELECTION == 2:
            error = abs(self.current_speed - self.ref_speed)
            reward = -np.exp(error)

        # 4. Thresholded absolute error
        elif ERROR_SELECTION == 3:
            error = abs(self.current_speed - self.ref_speed)
            reward = 1.0 if error < 0.5 else -error

        # 5. Cubed error
        elif ERROR_SELECTION == 4:
            error = abs(self.current_speed - self.ref_speed)
            reward = -error**3

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: Dependent on loss function called
    """

    def __init__(self, full_data, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        self.idx = 0
        self.current_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        ref_speed = self.full_data[self.idx]
        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        ref_speed = self.full_data[self.idx]
        
        # 1: Absolute error
        if ERROR_SELECTION == 0:
            error = abs(self.current_speed - ref_speed)
            reward = -error

        # 2: Squared error
        elif ERROR_SELECTION == 1:
            error = self.current_speed - ref_speed
            reward = -error**2

        # 3: Exponential error
        elif ERROR_SELECTION == 2:
            error = abs(self.current_speed - ref_speed)
            reward = -np.exp(error)

        # 4. Thresholded absolute error
        elif ERROR_SELECTION == 3:
            error = abs(self.current_speed - ref_speed)
            reward = 1.0 if error < 0.5 else -error

        # 5. Cubed error
        elif ERROR_SELECTION == 4:
            error = abs(self.current_speed - ref_speed)
            reward = -error**3

        else:
            error = -1000.0
            reward = -1000.0

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="RL Model Index (0-3)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (Default = 3e-4)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (Default = 256)"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=200000,
        help="Buffer size (Default = 200000)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Tau (Default = 0.005)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Gamma (Default = 0.99)"
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=-1.0,
        help="Gamma (Auto = -1.0, Rest = Float)"
    )
    parser.add_argument(
        "--net_arch",
        type=str,
        default="256,256",
        help="Net Arch defined as 2 integers (Default = 256,256)"
    )
    parser.add_argument(
        "--reward",
        type=int,
        default=0,
        help="Reward function. 0 = Absolute error, 1 = Squared error, 2 = Exponential error, 3 = Thresholded absolute error, 4 = Cubed error."
    )
    args = parser.parse_args()
    ent_coef_param = 'auto' if args.ent_coef == -1.0 else args.ent_coef
    net_arch_param = list(map(int, args.net_arch.split(",")))
    ERROR_SELECTION = args.reward

    log_dir = os.path.join(
        args.output_dir,
        f"model-{args.model}_chunk-{args.chunk_size}_lr-{args.learning_rate}_batch-{args.batch_size}_buffer-{args.buffer_size}_tau-{args.tau}_gamma-{args.gamma}_ent-{ent_coef_param}_arch-{'-'.join(map(str, net_arch_param))}_reward-{args.reward}"
    )

    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    chunk_size = args.chunk_size

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, delta_t=1.0)

    train_env = DummyVecEnv([make_train_env])

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=net_arch_param, activation_fn=nn.ReLU)

    """
    Pass in model type from command line
    0. SAC (Default)
    1. PPO
    2. TD3
    3. DDPG
    """

    # Model 1: SAC
    if args.model == 0:
        model = SAC(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            ent_coef=ent_coef_param,
            device=device                   # Do not change
        )
        model_str = "SAC"
    
    # Model 2: PPO
    elif args.model == 1:
        model = PPO(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=ent_coef_param,
            device=device                   # Do not change
        )
        model_str = "PPO"

    # Model 3: TD3
    elif args.model == 2:
        model = TD3(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            device=device                   # Do not change
        )
        model_str = "TD3"

    # Model 4: DDPG
    elif args.model == 3:
        model = DDPG(
            policy="MlpPolicy",             # Do not change
            env=train_env,                  # Do not change
            verbose=1,                      # Do not change
            policy_kwargs=policy_kwargs,    # Do not change
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            tau=args.tau,
            gamma=args.gamma,
            device=device                   # Do not change
        )
        model_str = "DDPG"
    
    # Invalid model number
    else:
        print(f"Invalid model code: {args.model}. Aborting code.")
        exit(1)

    # Print the current hyperparameter configuration
    print("\n\n----- Hyperparameter Configuration -----")
    print(f"Model = {model_str}")
    print(f"Chunk size = {chunk_size}")
    print(f"Learning rate = {args.learning_rate}")
    print(f"Batch size = {args.batch_size}")
    print(f"Buffer size = {args.buffer_size}")
    print(f"Tau = {args.tau}")
    print(f"Gamma = {args.gamma}")
    print(f"Entropy Coefficient = {ent_coef_param}")
    print(f"Net Arch = {args.net_arch}")
    print(f"Reward Function = {args.reward}")
    print("----------------------------------------\n")

    model.set_logger(logger)

    total_timesteps = 100_000
    callback = CustomLoggingCallback(log_dir)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"sac_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    avg_test_reward = np.mean(rewards)
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")

    # Calculate performance metrics
    mae = mean_absolute_error(reference_speeds, predicted_speeds)
    mse = mean_squared_error(reference_speeds, predicted_speeds)
    rmse = np.sqrt(mse)
    r2 = r2_score(reference_speeds, predicted_speeds)

    # Plot the entire test
    fig_path = os.path.join(log_dir, f"test_plot_chunk{chunk_size}.png")
    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on full 1200-step dataset (chunk_size={chunk_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved test plot to {fig_path}")

    # Save the test results to a txt file
    output_file = os.path.join(log_dir, "test_results.txt")
    with open(output_file, "w") as f:
        f.write("----- Hyperparameter Configuration -----\n")
        f.write(f"Model = {model_str}\n")
        f.write(f"Chunk size = {chunk_size}\n")
        f.write(f"Learning rate = {args.learning_rate}\n")
        f.write(f"Batch size = {args.batch_size}\n")
        f.write(f"Buffer size = {args.buffer_size}\n")
        f.write(f"Tau = {args.tau}\n")
        f.write(f"Gamma = {args.gamma}\n")
        f.write(f"Entropy Coefficient = {ent_coef_param}\n")
        f.write(f"Net Arch = {args.net_arch}\n")
        f.write(f"Reward Function = {args.reward}\n")
        f.write("----------------------------------------\n\n")

        f.write("Timestep\tPredicted Speed\tReference Speed\tReward\n")
        f.write("------------------------------------------------------\n")
        for i in range(len(predicted_speeds)):
            f.write(f"{i+1}\t\t{predicted_speeds[i]:.3f}\t\t{reference_speeds[i]:.3f}\t\t{rewards[i]:.3f}\n")

        f.write("\n----------------------------------------\n")
        f.write(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}\n")
        f.write(f"[TEST] Mean Absolute Error (MAE): {mae:.3f}\n")
        f.write(f"[TEST] Mean Squared Error (MSE): {mse:.3f}\n")
        f.write(f"[TEST] Root Mean Squared Error (RMSE): {rmse:.3f}\n")
        f.write(f"[TEST] R2 Score (R2): {r2:.3f}\n")
        f.write("----------------------------------------\n")
    print(f"Saved test results to {output_file}")


    # Load rewards from training log and plot
    reward_log_file = os.path.join(log_dir, "training_log.csv")
    if os.path.exists(reward_log_file):
        df_rewards = pd.read_csv(reward_log_file)
        plt.figure(figsize=(10, 5))
        plt.plot(df_rewards["timestep"], df_rewards["average_reward"], label="Training Reward", linestyle="-")
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.title("Training Convergence")
        plt.legend()
        plt.savefig(os.path.join(log_dir, "training_convergence.png"))
        plt.close()
        print(f"Saved training convergence plot to {log_dir}/training_convergence.png")


if __name__ == "__main__":
    main()
