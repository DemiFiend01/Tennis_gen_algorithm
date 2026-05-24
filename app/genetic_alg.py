
'''To Do:
- Add Selection methods:
- Choose one winner and select the best state.
'''

class GeneticAlgorithm:
    def __init__(self, no_inv: int, no_gen):
        """Initialises the genetic algorithm

        Args:
            no_inv (int): Number of individuals
            no_gen (int): Number of generations
        """
        self.no_inv = no_inv
        self.no_gen = no_gen

    def set_game_info(self):
        #call from main? then call run here. in main call input = uhh output from fdhgkjhgkjhkjdghs i dont know,
        #maybe we should move the atari stuff from main into here? lets say we run 1000 or 500 steps per generation?
        # without video for speed up, then we could just see the best individual from a generation? and save their results in a file
        pass