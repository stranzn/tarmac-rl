from stable_baselines3 import PPO
from car_env import CarEnv

env = CarEnv(render_mode="human")

# load the trained model
model = PPO.load("car_ppo", env=env)

print("=== Testing Trained Model ===")
print("Watching the best version the AI learned...\n")

obs, _ = env.reset()
total_reward = 0
episodes = 0

try:
    while True:  # run until you close the window
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        total_reward += reward
        
        if done:
            episodes += 1
            print(f"Episode {episodes} finished | Total Reward: {total_reward:.1f}")
            obs, _ = env.reset()
            total_reward = 0

except KeyboardInterrupt:
    print("\nTest stopped by user.")
except Exception as e:
    print(f"Error: {e}")
finally:
    env.close()

print("Test finished.")