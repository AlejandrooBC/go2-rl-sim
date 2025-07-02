from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
#carpole
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env('CartPole-v1', n_envs=1, seed=42)
model = PPO("MlpPolicy", vec_env, verbose=1,device='cpu')

model.policy.pre