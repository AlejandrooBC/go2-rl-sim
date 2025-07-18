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
        if "height_delta" in info:
            self.logger.record("custom/height_delta", info["height_delta"])
        if "height_delta_penalty" in info:
            self.logger.record("custom/height_delta_penalty", info["height_delta_penalty"])
        if "lateral_position" in info:
            self.logger.record("custom/lateral_position", info["lateral_position"])
        if "lateral_penalty" in info:
            self.logger.record("custom/lateral_penalty", info["lateral_penalty"])
        if "reward" in info:
            self.logger.record("custom/reward", info["reward"])

        return True

# Initialize the sim environment and check its compatability
env = UnitreeGo2Env()
check_env(env, warn=True)

# Load fine-tuned model and attach the environment
model = PPO.load("trained_models_single/ppo_go2_gallop_v5", env=env)
model.set_env(env)

# Setting a new learning rate
model.lr_schedule = lambda _: 1e-5
model.policy.optimizer = model.policy.optimizer_class(
    model.policy.parameters(),
    lr=1e-5,
    **model.policy.optimizer_kwargs,
)

# Set up checkpoint saving
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./trained_models_single/",
    name_prefix=f"{model_name}_checkpoint_",
    save_replay_buffer=False,
    save_vecnormalize=False
)

# Fine-tune the model
model.learn(
    total_timesteps=500_000, # Fine-tuning timesteps
    tb_log_name=f"run_finetune_{timestamp}",
    callback=[TensorboardCallback(), checkpoint_callback],
    reset_num_timesteps=False # IMPORTANT: Continue from the most recent/best fined-tuned model training
)

# Save the final fine-tuned model
model.save(f"trained_models_single/{model_name}")
print(f"Fine-tuning complete. Model saved as '{model_name}.zip'")