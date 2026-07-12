import random 
import os 
import numpy as np 
import torch 
import matplotlib.pyplot as plt

from src.game import LunarLanderEnv
from src.agent import Agent

##############################################
############## Reproducibility ###############
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

##############################################
############## Hyperparameters ###############
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000

HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
GAMMA = 0.99

BUFFER_CAPACITY = 100_000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_EVERY = 10 # episodes 

WEIGHTS_PATH = "weights.pth"
REWARD_PLOT_PATH = "training_rewards.png"

def moving_average(values, window_size=50):
    if len(values) < window_size:
        return []

    averages = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        averages.append(sum(window) / window_size)

    return averages

def plot_rewards(episode_rewards, save_path):
    plt.figure(figsize=(10, 5))

    plt.plot(episode_rewards, label="Episode reward")

    ma_rewards = moving_average(episode_rewards, window_size=50)
    if len(ma_rewards) > 0:
        plt.plot(
            range(49, 49 + len(ma_rewards)),
            ma_rewards,
            label="Moving average (50 episodes)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train():
    env = LunarLanderEnv(render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE 
    )

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Device:", agent.device)

    epsilon = EPSILON_START

    episode_rewards = []
    episode_losses = []

    best_moving_avg = -float("inf")

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset(seed=SEED + episode)

        total_reward = 0
        losses = []

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, epsilon)

            next_state, reward, done = env.step(action)

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state 
            total_reward += reward

            if done: 
                break 

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)

        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        episode_losses.append(avg_loss)


        recent_rewards = episode_rewards[-50:]
        moving_avg_reward = np.mean(recent_rewards)

        if moving_avg_reward > best_moving_avg and episode >= 50:
            best_moving_avg = moving_avg_reward
            agent.save(WEIGHTS_PATH)

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg50: {moving_avg_reward:8.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Loss: {avg_loss:.5f}"
            )

    if not os.path.exists(WEIGHTS_PATH):
        agent.save(WEIGHTS_PATH)

    plot_rewards(episode_rewards, REWARD_PLOT_PATH)

    env.close()

    print("\nTraining finished.")
    print(f"Weights saved to: {WEIGHTS_PATH}")
    print(f"Reward plot saved to: {REWARD_PLOT_PATH}")

if __name__ == "__main__": 
    train()
