import numpy as np

''' Input layer:
enemy_x
enemy_y
player_x
player_y
ball_x
ball_y
player_score
enemy_score

// To be calculated
ball_v
ball_dir_x
ball_dir_y
'''

'''To Do:
- Fitness function
- Random initialization
- Some forward function
- Choosing of one state or not here? Could be in the genetic algorithm
'''

class Individual:
    def __init__(self, state, n_h1: int = 32, n_h2: int = 16, n_y: int = 18):
      """Initializes an individual in a neural network.

      Args:
          state (_type_): Inputs in form of info from RAM and calculated values.
          n_h1 (int): Number of nodes in hidden layer 1
          n_h2 (int): Number of nodes in hidden layer 2
          n_y (int): Number of output nodes aka game inputs. Defaults to 18.
      """
      self.fitness = 0
      # Inputs
      self.state = state
      # Nodes in the input layer
      self.n_x = len(state)
      self.n_h1 = n_h1
      self.n_h2 = n_h2
      self.n_y = n_y

      # Weights matrices initialization with the Xavier init.
      xavier_init = np.sqrt(1/self.n_x)
      self.W1 = np.random.randn(self.n_x, self.n_h1) * xavier_init
      self.W2 = np.random.randn(self.n_h1, self.n_h2) * xavier_init
      self.W3 = np.random.randn(self.n_h2, self.n_y) * xavier_init

    def forward(self):
       pass

    def eval_fitness(self):
       pass
    
    def select_game_input(self):
       # Returns 1 input from 18 available game inputs
       pass

    # For mutation
    def get_weights(self):
        pass

    def set_weights(self):
       pass