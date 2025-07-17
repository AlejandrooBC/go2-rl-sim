import time
from stable_baselines3 import PPO
from go2_env_vec import UnitreeGo2Env
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv

# Load environment and normalization wrapper
dummy_env = DummyVecEnv([lambda: UnitreeGo2Env(render_mode="human")])
eval_env = VecNormalize.load("vecstats/vecnormalize.pkl", dummy_env)

# Disable updates to stats during evaluation
eval_env.training = False
eval_env.norm_reward = False

# Load the trained model
model = PPO.load("trained_models_vec/ppo_go2_vec_20250716-200521_checkpoint__2000000_steps")  # Update filename

# Create a writer for TensorBoard logs
writer = SummaryWriter(log_dir="tensorboard/eval")

# Number of episodes to evaluate the policy on
n_eval_episodes = 100

# Loop through evaluation episodes
for ep in range(n_eval_episodes):
    obs, _ = eval_env.reset() # Reset environment to start state
    done = False
    ep_reward = 0
    steps = 0

    # Run the policy until the episode ends or max steps reached
    while not done and steps < 9000:
        # Predict the next action using the trained policy (no randomness)
        action, _ = model.predict(obs, deterministic=True)

        # Step through the environment using the action
        obs, reward, terminated, truncated, _ = eval_env.step(action)

        # Render the robot from the environment
        eval_env.render()
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