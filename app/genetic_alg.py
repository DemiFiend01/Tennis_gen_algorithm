
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