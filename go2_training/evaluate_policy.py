import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from go2_env import UnitreeGo2Env
from torch.utils.tensorboard import SummaryWriter

# Create a writer for TensorBoard logs
writer = SummaryWriter(log_dir="tensorboard/eval")

# Create and wrap the evaluation environment in DummyVecEnv
eval_env = DummyVecEnv([lambda: UnitreeGo2Env(render_mode="human")])

# Load the VecNormalize stats from training
eval_env = VecNormalize.load("vecstats/{model_name}_vecnormalize.pkl", eval_env)

# Set to evaluation mode (disable running stats updates, disable reward normalization)
eval_env.training = False
eval_env.norm_reward = False

# Load the trained PPO model
model = PPO.load("trained_models/ppo_go2_20250707-152337", env=eval_env)

# Number of episodes to evaluate the policy on
n_eval_episodes = 100

# Loop through evaluation episodes
for ep in range(n_eval_episodes):
    obs = eval_env.reset() # Reset environment to start state
    done = False
    ep_reward = 0
    steps = 0

    # Run the policy until the episode ends or max steps reached
    while not done and steps < 9000:
        # Predict the next action using the trained policy (no randomness)
        action, _ = model.predict(obs, deterministic=True)

        # Step through the environment using the action
        obs, reward, terminated, truncated, _ = eval_env.step(action)

        # Render the robot from the underlying environment
        eval_env.envs[0].render()
        time.sleep(0.03)

        # Check for the episode's end
        done = terminated or truncated

        # Accumulate reward and count steps
        ep_reward += reward[0] # Reward is a numpy array of shape (1,) due to VecEnv
        steps += 1

    # Print and log reward to Tensorboard
    print(f"Episode {ep + 1} reward: {ep_reward}")
    writer.add_scalar("eval/episode_reward", ep_reward, ep)

# Close the Tensorboard writer
writer.close()