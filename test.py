import torch as th
import gym_torcs
import gym
import sys
import os

from stable_baselines3 import A2C

#env = gym.make('CartPole-v1')
env = gym_torcs.TorcsEnv()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])

if os.path.exists('a2c_torcs.zip'):
  print('Using saved model')
  model = A2C.load('a2c_torcs', env=env)
else:
  model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=1000, log_interval=5)
print('Learning done')
model.save('a2c_torcs')
env.end()
sys.exit()
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()