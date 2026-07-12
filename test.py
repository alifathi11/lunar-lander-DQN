import torch 

from game import LunarLanderEnv
from agent import Agent

WEIGHTS_PATH = "weights.pth"

SEED = 42

EPISODES = 10

def test(): 
    env = LunarLanderEnv(render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    agent.load(WEIGHTS_PATH)

    agent.policy_net.eval()

    for episode in range(1, EPISODES + 1):
        state = env.reset(SEED + episode)

        total_reward = 0
        done = False

        print(f"\n--- Episode {episode} ---")

        while not done:
            action = agent.select_action(
                state=state, 
                eps=0.0
            )

            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward

        print(
            f"Total reward: {total_reward:.2f}"
        )

    env.close()

if __name__ == "__main__":
    test()