from atariari.benchmark.wrapper import AtariARIWrapper
import atariari

class FixedAtariWrapper(AtariARIWrapper):
    """AtariARIWrapper wrapper due to ALE and wrapper version conflict.

    Contains simple methods, both overriden and not.

    Attributes:
        env: environment
    """
    def __init__(self, env):
        super(AtariARIWrapper, self).__init__(env)
        self.ram_annotations = atariari.benchmark.ram_annotations.atari_dict["tennis"]

    def get_labels(self, ram):
        # Extracting the important named values from RAM
        return {label: int(ram[idx]) for label, idx in self.ram_annotations.items()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self.env.unwrapped.ale.getRAM()
        label = self.get_labels(info)
        return obs, reward, terminated, truncated, {"labels": label, "ram": info}
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
