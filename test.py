import torch as th
import gym_torcs
import gym
import sys
import os

from stable_baselines3 import A2C

#env = gym.make('CartPole-v1')
env = gym_torcs.TorcsEnv()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[128, dict(pi=[64], vf=[256])])

if os.path.exists('a2c_torcs.zip'):
  print('Using saved model')
  model = A2C.load('a2c_torcs', env=env)
else:
  model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tboard_logs/")

for i in range(10):
  print('Run {}: start'.format(i))
  model.learn(total_timesteps=1000, tb_log_name="run{}".format)
  print('Run {}: end'.format(i))
model.save('a2c_torcs')
print('Learning done')
env.end()
sys.exit()
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()