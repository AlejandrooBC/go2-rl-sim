import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from go2_env_single import UnitreeGo2Env

# Generate a timestamp string to uniquely identify this training run
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"ppo_go2_{timestamp}"

# Optional: Custom callback to log custom info during training
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
        if "delta_x" in info:
            self.logger.record("custom/delta_x", info["delta_x"])
        if "steps_alive" in info:
            self.logger.record("custom/steps_alive", info["steps_alive"])
        if "reward" in info:
            self.logger.record("custom/reward", info["reward"])

        return True

# Initialize the sim environment and check its compatability
env = UnitreeGo2Env()
check_env(env, warn=True)

# Create the PPO model
model = PPO(
    "MlpPolicy", # Use a multi-layer perceptron (MLP) policy
    env, # Custom Go2 environment
    verbose=1, # Print training progress to terminal
    tensorboard_log="./ppo_go2_tensorboard/", # Enable Tensorboard logging (path to save logs)
    n_steps=8192,
    batch_size=1024,
    n_epochs=5,
    device="cuda"
)


# Checkpoint saving every 1 million steps
checkpoint_callback = CheckpointCallback(
    save_freq=2_000_000,
    save_path="./trained_models_single/",
    name_prefix=f"{model_name}_checkpoint_",
    save_replay_buffer=False,
    save_vecnormalize=False
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
    total_timesteps=25_000_000, # Number of training timesteps
    tb_log_name=f"run_{timestamp}", # Folder name of this run's logs
    callback=[TensorboardCallback(), checkpoint_callback] # Log step count to Tensorboard, log checkpoints
)

# Save the model with a unique timestamped filename
model.save(f"trained_models_single/{model_name}")
print(f"Training complete. Model saved as '{model_name}.zip'")