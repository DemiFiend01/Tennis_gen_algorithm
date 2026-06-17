import FixedAtariWrapper as faw
import ale_py
import gymnasium as gym
from time import sleep

from tutorial import Tutorial
from user import tennis_playable
from genetic_alg import GeneticAlgorithm

def test():
    # Using wrapper to a wrapper due to version differences
    env = faw.FixedAtariWrapper(gym.make('ALE/Tennis-v5', render_mode="human", mode=2))
    
    # env_u = env.unwrapped
    # print(dir(env_u))  # look for setMode, game, rom, ale, _env etc.
    # print(type(env_u))
    # print(env.unwrapped._game_mode)
    # env.unwrapped.ale.setMode(2)
    # print(env.unwrapped.get_action_meanings())
    obs, info = env.reset()
    
    # env.unwrapped.ale.setMode(2)
    # print(env.unwrapped.get_action_meanings())
    # print(env.unwrapped._game_mode)
    # print(dir(env.unwrapped.ale))
    # help(env.unwrapped.ale.act)
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(1)
        labels = info["labels"]
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {labels}")
        # p1_action = env.action_space.sample()
        # p2_action = env.action_space.sample()
        # env.unwrapped.ale.act(p1_action, p2_action)
        # obs = env.unwrapped._get_obs()
        # info = env.unwrapped._get_info()
        # print(info)
        # sleep(0.2)
    env.close()

if __name__ == '__main__':
    genAlg = GeneticAlgorithm(no_inv=60, no_gen=60, threads=1, parents_frac=0.2)
    genAlg.run()

    genAlg.ga_instance.plot_fitness()
   
    #tutorial = Tutorial()
    #tutorial.run()
    #tennis_playable()

    #test()