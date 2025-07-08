from stable_baselines3 import PPO
from go2_env_single import UnitreeGo2Env
from torch.utils.tensorboard import SummaryWriter
import time

# Create a writer for TensorBoard logs
writer = SummaryWriter(log_dir="tensorboard/eval")

# Load the custom environment and the trained PPO model
env = UnitreeGo2Env()
model = PPO.load("trained_models_single/ppo_go2_20250708-121925") # Replace this with the correct model's name

# Number of episodes to evaluate the policy on
n_eval_episodes = 100

# Loop through evaluation episodes
for ep in range(n_eval_episodes):
    obs, _ = env.reset() # Reset environment to start state
    done = False
    ep_reward = 0
    steps = 0

    # Run the policy until the episode ends or max steps reached
    while not done and steps < 9000:
        # Predict the next action using the trained policy (no randomness)
        action, _ = model.predict(obs, deterministic=True)

        # Step through the environment using the action
        obs, reward, terminated, truncated, _ = env.step(action)

        # Render the robot in a viewer window
        env.render()
        time.sleep(0.03)

        # Check for the episode's end
        done = terminated or truncated

        # Accumulate reward and count steps
        ep_reward += reward
        steps += 1

    # Print and log reward to Tensorboard
    print(f"Episode {ep + 1} reward: {ep_reward}")
    writer.add_scalar("eval/episode_reward", ep_reward, ep)

# Close the Tensorboard writer
writer.close()