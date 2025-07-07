import time
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from go2_env import UnitreeGo2Env

# Number of vectorized/parallel environments (train the agent on N environments per step)
NUM_ENVS = 8

# Factory for creating environments (required by SubprocVecEnv)
def make_env():
    def _init():
        env = UnitreeGo2Env(render_mode="none")
        return env
    return _init

# Custom callback to log custom info during training
class TensorboardCallback(BaseCallback):
    # Called every time the model takes a step --> used to log custom metrics to Tensorboard
    def _on_step(self) -> bool:
        self.logger.record("custom/step_count", self.num_timesteps)
        info = self.locals.get("infos", [{}])[0]

        if "x_position" in info:
            self.logger.record("custom/x_position", info["x_position"])
        if "z_height" in info:
            self.logger.record("custom/z_height", info["z_height"])
        if "x_velocity" in info:
            self.logger.record("custom/x_velocity", info["x_velocity"])
        if "reward" in info:
            self.logger.record("custom/reward", info["reward"])

        return True

# Main block - required for SubprocVecEnv to work without forking issues
if __name__ == "__main__":
    mp.set_start_method("fork", force=True) # Ensures Linux uses "fork"

    # Check single environment compatibility once
    check_env(UnitreeGo2Env(), warn=True)

    # Create the vectorized environment using multiple subprocesses
    vec_env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Generate a timestamp string to uniquely identify this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"ppo_go2_{timestamp}"

    # Create the PPO model
    model = PPO(
        "MlpPolicy", # Use a multi-layer perceptron (MLP) policy
        vec_env, # Custom Go2 environment
        verbose=1, # Print training progress to terminal
        tensorboard_log="./ppo_go2_tensorboard/", # Enable Tensorboard logging (path to save logs)
        n_steps=2048 * NUM_ENVS, # Number of steps that are collected from each environment per update cycle
        batch_size=2048, # Number of samples (state, action, reward, next state) processed during each gradient update
        n_epochs=10, # Number of complete passes of the entire training dataset through the RL algorithm
        device="cuda", # GPU usage
        policy_kwargs=dict(net_arch=[256, 256]) # Simpler than default [64, 64]
    )

    """
    Train the model using PPO:
    1. PPO is an on-policy RL algorithm.
    - It collects fresh data using the current policy to update itself.
    - Improves sample quality, but it can't reuse old data.
    2. PPO uses gradient ascent to update the policy's weights.
    3. PPO trains a neural network to output actions (12D control vector)
    """
    model.learn(
        total_timesteps=8_000_000, # Number of training timesteps
        tb_log_name=f"run_{timestamp}", # Folder name of this run's logs
        callback=TensorboardCallback() # Log step count to Tensorboard
    )

    # Save the model with a unique timestamped filename
    model.save(f"trained_models/{model_name}")
    print(f"Training complete. Model saved as '{model_name}.zip'")