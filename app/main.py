import FixedAtariWrapper as faw
import ale_py
import gymnasium as gym
from time import sleep

from tutorial import Tutorial
from user import tennis_playable

def test():
    # Using wrapper to a wrapper due to version differences
    env = faw.FixedAtariWrapper(gym.make('ALE/Tennis-v5', render_mode="human"))
    obs, info = env.reset()
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        labels = info["labels"]
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {labels}")
        sleep(0.2)
    env.close()

if __name__ == '__main__':
    #tutorial = Tutorial()
    #tutorial.run()
    tennis_playable()

    #test()