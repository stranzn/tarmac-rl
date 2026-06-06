import argparse
from stable_baselines3 import PPO
from car_env import CarEnv

# initialises the environment and PPO model, then runs training
def main(render_mode):
    env = CarEnv(render_mode=render_mode)
    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,   # step size for gradient updates; 3e-4 is a safe default for PPO
        n_steps=2048,         # steps collected per environment per update; larger = more stable but slower
        batch_size=64,        # minibatch size used during each gradient update
        n_epochs=10,          # how many times to reuse each collected batch before discarding it
        gamma=0.99,           # discount factor; high value means the agent cares about long-term rewards
        gae_lambda=0.95,      # GAE smoothing — trades off bias vs variance in advantage estimates
        clip_range=0.2,       # PPO clipping threshold; prevents policy updates from being too large
        ent_coef=0.01,        # entropy bonus weight; encourages exploration by penalising certainty
        vf_coef=0.5,          # value function loss weight relative to policy loss
        max_grad_norm=0.5,    # gradient clipping; prevents exploding gradients during updates
    )

    print("=== Training Started ===") 
    if render_mode == "human":
        print("Watch the car learning in the window...\n")
    else:
            print("Training without GUI...\n")

    try:
        model.learn(total_timesteps=500000,
                        callback=lambda locals_, globals_: (env.render(), False) if render_mode == "human" else (False, False))
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

    model.save("car_ppo")

# clear alert when training finishes
print("\n" + "="*50)
print("TRAINING FINISHED SUCCESSFULLY!")
print("Model saved as 'car_ppo'")
print("Run test.py to watch the trained agent.")
print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO model for the CarEnv.")
    parser.add_argument('--gui', action='store_true', help='Run with GUI')
    args = parser.parse_args()
    render_mode = "human" if args.gui else None
    main(render_mode)