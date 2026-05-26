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

      #Extract the actual state values without labels, but in the genetic_algorithm
      #because right now we have {'enemy_x': 62, 'enemy_y': 2, 'enemy_score': 0, 'ball_x': 71, 'ball_y': 33, 'player_x': 64, 'player_y': 142, 'player_score': 0}

      # Weights matrices initialization with the Xavier init.
      xavier_init = np.sqrt(1/self.n_x)
      self.W1 = np.random.randn(self.n_x, self.n_h1) * xavier_init
      self.W2 = np.random.randn(self.n_h1, self.n_h2) * xavier_init
      self.W3 = np.random.randn(self.n_h2, self.n_y) * xavier_init

    def forward(self, new_state):
       self.state = new_state
       # Inputs vector multiplication by the first weights matrix
       h1 = np.maximum(0, self.state @ self.W1 ) # ReLU
       # First hidden layer matrix multiplication by the second weights matrix
       h2 = np.maximum(0, h1 @ self.W2) # ReLU

       # Use softmax as it is best for multivalue outputs 
       x = h2 @ self.W3
       e_x = np.exp(x - np.max(x))
       softmax = e_x / e_x.sum()
       return int(np.argmax(softmax))

    # For mutation
    def get_weights(self):
        # Flatten into one array
        return np.concatenate([self.W1.ravel(), self.W2.ravel(), self.W3.ravel()])

    def set_weights(self, flat: np.ndarray):
       i = 0
       for name, shape in [('W1', (self.n_x, self.n_h1)),
                           ('W2', (self.n_h1, self.n_h2)),
                           ('W3', (self.n_h2, self.n_y))]:
          size = shape[0] * shape[1]
          setattr(self, name, flat[i:i + size].reshape(shape))
          i += size