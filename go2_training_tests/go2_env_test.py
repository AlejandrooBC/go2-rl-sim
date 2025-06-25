import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
from gymnasium import spaces

# Custom Gymnasium environment for simulating the Unitree Go2 robot using MuJoCo
class UnitreeGo2Env(gym.Env):
    def __init__(self, model_path="../mujoco_menagerie/unitree_go2/go2.xml"):
        # Load the MuJoCo model and allocate simulation data
        self.model = MjModel.from_xml_path(model_path) # Blueprint of the Go2 model (geometry, actuators/motors, etc.)
        self.data = MjData(self.model) # Live state of the Go2 and world (qpos, qvel, ctrl, etc.) - snapshot

        # Number of MuJoCo physics steps to take per environment step - for each call to step()
        self.sim_steps = 30

        # Define the action space: one control input per actuator (what the agent can output)
        # If the shape = (12,) then the policy will output 12 numbers between (low, high) per step
        self.action_space = spaces.Box( # Box = continuous values with (low, high) bounds
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,), # nu = number of actuators
            dtype=np.float32)

        # Define the observation space: joint positions and velocities (what the agent sees)
        # The policy receives a vector of all joint angles, base pose, and their velocities
        obs_dim = self.model.nq + self.model.nv # nq = positions, nv = velocities
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,), # nq + nv
            dtype=np.float32)

        # Counter for throttled printing
        self.print_counter = 0

    # The core loop of the environment - each time our PPO agent sends an action, this happens:
    def step(self, action):
        # Apply the control signal (input) to the robot - array of size 12 (number of actuators on the Go2)
        # The Go2 has 4 legs, 3 actuators per leg (abduction, hip, knee) = 12 total actuators
        self.data.ctrl[:] = action

        # Step the physics forward several times for smoother simulation
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Construct the updated observation: [qpos, qvel]
        obs = np.concatenate([self.data.qpos, self.data.qvel])

        # Calculating torque effort
        torque_effort = np.sum(np.square(self.data.ctrl))

        # Define forward velocity
        forward_velocity = self.data.qvel[0]

        # Penalize falling outside a height range
        # height = self.data.qpos[2]
        # min_height = 0.20
        # target_height = 0.24
        # max_height = 0.30
        # if height < min_height or height > max_height:
        #     height_penalty = np.abs(height - target_height)
        # else:
        #     height_penalty = 0.0

        # Or soft quadratic height penalty
        # height_penalty = (height - target_height) ** 2

        reward = 20 * forward_velocity
        # reward = (
        #     5.0 * forward_velocity
        #     -0.5 * height_penalty
        #     -0.01 * torque_effort
        # )

        # Placeholder termination flags
        terminated = bool(False) # (self.data.qpos[2] < 0.2)
        truncated = bool(False)

        # Info for Tensorboard logging
        info = {
            "x_position": self.data.qpos[0],
            "z_height": self.data.qpos[2],
            "x_velocity": self.data.qvel[0],
            "reward": reward
        }

        # Print metrics every 100 environment steps
        # if self.print_counter % 100 == 0:
        #     print(
        #         f"X_POSITION: {self.data.qpos[0]:.3f} | "
        #         f"Z_HEIGHT: {self.data.qpos[2]:.3f} | "
        #         f"X_VELOCITY: {self.data.qvel[0]:.3f} | "
        #         f"REWARD: {reward:.3f}"
        #     )
        # self.print_counter += 1

        return obs.astype(np.float32), reward, terminated, truncated, info

    # Reset sim to a known starting state (called at the beginning of each episode during training/evaluation)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Calls parent class's reset() method
        mujoco.mj_resetData(self.model, self.data) # Reset old sim data (sets qpos, qvel, etc. to defaults/zeros)

        # Set initial robot pose for the Go2 based on go2.xml
        self.data.qpos[:] = np.array([
            0, 0, 0.27, # Base pos (x = 0, y = 0, z = 0.27 m) - slightly above the floor
            1, 0, 0, 0, # Base orientation quaternion (w, x, y, z) --> robot is upright with no rotation

            # Joint angles for each leg: abduction (side swing), hip (forward/back), knee (extension/flexion)
            0, 0.9, -1.8, # Front-left
            0, 0.9, -1.8, # Front-right
            0, 0.9, -1.8, # Rear-left
            0, 0.9, -1.8 # Rear-right
        ])

        # Set all joint and base velocities to zero - robot not moving at spawn
        self.data.qvel[:] = np.zeros_like(self.data.qvel)

        # Tells MuJoCo to re-calculate everything (kinematics, contacts, etc.) after you manually change the state
        mujoco.mj_forward(self.model, self.data)

        # Packages the observations as a [positions + velocities] array and returns it
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        self.print_counter = 0 # Reset counter each episode
        return obs.astype(np.float32), {}

    # Renders the MuJoCo environment for model visualization
    def render(self, mode="human"):
        # Launch the passive viewer only once
        if not hasattr(self, "viewer"):
            self.viewer = viewer.launch_passive(self.model, self.data)

        # Update the viewer with the latest state
        self.viewer.sync()