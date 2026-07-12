import time
from game import LunarLanderEnv

def run_random_agent():
    env = LunarLanderEnv(render_mode="human")
    
    episodes = 10
    
    for ep in range(episodes):
        state = env.reset(seed=42 + ep)
        total_reward = 0
        done = False
        
        print(f"--- Starting Episode {ep + 1} ---")
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done = env.step(action)
            total_reward += reward
            time.sleep(0.02)
            
        print(f"Episode {ep + 1} finished with Total Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    run_random_agent()