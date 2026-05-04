import gymnasium as gym
import ale_py

def test():
    env = gym.make('ALE/Tennis', render_mode="human")
    obs, info = env.reset()
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

if __name__ == '__main__':
    test()