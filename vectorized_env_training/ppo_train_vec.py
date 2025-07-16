import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from go2_env_vec import UnitreeGo2Env

# Timestamped model name
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"ppo_go2_vec_{timestamp}"

# Custom callback to log info during training
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
        if "z_velocity" in info:
            self.logger.record("custom/z_velocity", info["z_velocity"])
        if "delta_x" in info:
            self.logger.record("custom/delta_x", info["delta_x"])
        if "steps_alive" in info:
            self.logger.record("custom/steps_alive", info["steps_alive"])
        if "reward" in info:
            self.logger.record("custom/reward", info["reward"])

        return True

# Wrap the environment creation for each subprocess
def make_env():
    def _init():
        return UnitreeGo2Env(render_mode=None) # Disable rendering in parallel training
    return _init

if __name__ == "__main__":
    # Create vectorized environment with n_envs (subprocesses)
    n_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_go2_tensorboard/",
        n_steps=8192,
        batch_size=1024,
        n_epochs=5,
        device="cuda"
    )

    # Save model every 2M total timesteps = 2 million / 4 per subprocess
    checkpoint_callback = CheckpointCallback(
        save_freq=int(2_000_000 / n_envs),
        save_path="./trained_models_vec/",
        name_prefix=f"{model_name}_checkpoint_",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # Train the PPO agent
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=f"run_{timestamp}",
        callback=[TensorboardCallback(), checkpoint_callback]
    )

    # Save final model
    model.save(f"trained_models_vec/{model_name}")
    print(f"Training complete. Model saved as '{model_name}.zip'")