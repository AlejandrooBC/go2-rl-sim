import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from go2_env_single import UnitreeGo2Env

# Generate a timestamp string to uniquely identify this training run
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"ppo_go2_{timestamp}"

# Callback to log custom info during training
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
        if "lateral_position" in info:
            self.logger.record("custom/lateral_position", info["lateral_position"])
        if "r_linear_velocity" in info:
            self.logger.record("custom/r_linear_velocity", info["r_linear_velocity"])
        if "r_angular_velocity" in info:
            self.logger.record("custom/r_angular_velocity", info["r_angular_velocity"])
        if "r_height_penalty" in info:
            self.logger.record("custom/r_height_penalty", info["r_height_penalty"])
        if "r_pose_similarity" in info:
            self.logger.record("custom/r_pose_similarity", info["r_pose_similarity"])
        if "r_action_rate_penalty" in info:
            self.logger.record("custom/r_action_rate_penalty", info["r_action_rate_penalty"])
        if "r_vertical_velocity_penalty" in info:
            self.logger.record("custom/r_vertical_velocity_penalty", info["r_vertical_velocity_penalty"])
        if "r_orientation_penalty" in info:
            self.logger.record("custom/r_orientation_penalty", info["r_orientation_penalty"])
        return True

# Initialize the sim environment and check its compatability
check_env(UnitreeGo2Env(), warn=True)
env = UnitreeGo2Env()

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_go2_tensorboard/",
            learning_rate=0.0003,
            n_steps=8192,
            batch_size=1024,
            n_epochs=5,
            device="cuda")

# Checkpoint saving every user-defined number of steps
checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,
    save_path="./trained_models_single/",
    name_prefix=f"{model_name}_checkpoint_",
    save_replay_buffer=False,
    save_vecnormalize=False
)

# Train the PPO model
model.learn(
    total_timesteps=20_000_000, # Number of training timesteps
    tb_log_name=f"run_{timestamp}", # Folder name of this run's logs
    callback=[TensorboardCallback(), checkpoint_callback] # Log step count to Tensorboard, log checkpoints
)

# Save the model with a unique timestamped filename
model.save(f"trained_models_single/{model_name}")
print(f"Training complete. Model saved as '{model_name}.zip'")