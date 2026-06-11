import pygame
from stable_baselines3 import PPO
from car_env import CarEnv

pygame.init()  # initialise before CarEnv so the video system is ready

env = CarEnv(render_mode="human")

# load the trained model
model = PPO.load("car_ppo", env=env)
print("=== Testing Trained Model ===")
print("Watching the best version the AI learned...\n")
print("Controls: SPACE = pause/resume | ESC = quit\n")

obs, _       = env.reset()
total_reward = 0
episodes     = 0
paused       = False

try:
    while True:
        # handle input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused." if paused else "Resumed.")

        if paused:
            env.render()   # keep the window responsive while paused
            continue

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        total_reward += reward

        if done or truncated:
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