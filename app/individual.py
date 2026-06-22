import numpy as np
from pathlib import Path

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
    # MODIFIED state -> inputs_num
    def __init__(self, inputs_num : int, n_h1: int = 32, n_h2: int = 16, n_y: int = 18):
        """Initializes an individual in a neural network.

        Args:
            state      (_type_): (deprecated) Inputs in form of info from RAM and calculated values.
            inputs_num (int): Number of inputs to the NN.
            n_h1       (int): Number of nodes in hidden layer 1.
            n_h2       (int): Number of nodes in hidden layer 2.
            n_y        (int): Number of output nodes aka game inputs. Defaults to 18.
        """
        self.fitness = 0 # Remove?
        # Inputs
        # self.state = state # COMMENTED
        # Nodes in the input layer
        self.n_x  = inputs_num #len(state)
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.n_y  = n_y

        #Extract the actual state values without labels, but in the genetic_algorithm
        #because right now we have {'enemy_x': 62, 'enemy_y': 2, 'enemy_score': 0, 'ball_x': 71, 'ball_y': 33, 'player_x': 64, 'player_y': 142, 'player_score': 0}

        # Weights matrices initialization with the Xavier init.
        xavier_init = np.sqrt(1/self.n_x)
        self.W1 = np.random.randn(self.n_x, self.n_h1)  * xavier_init
        self.W2 = np.random.randn(self.n_h1, self.n_h2) * xavier_init
        self.W3 = np.random.randn(self.n_h2, self.n_y)  * xavier_init

    def from_npy(self, src : Path | str):
        ''' Creates an individual using data from given source file. '''

        if type(src) is str:
            src = Path(src).resolve()
            if not src.is_file():
                print(f"File doesn't exist: {src}")
                return None

        data = np.load(src, allow_pickle=True)

        n_1 = self.n_x * self.n_h1
        n_2 = n_1 + self.n_h1 * self.n_h2
        n_3 = n_2 + self.n_h2 * self.n_y

        self.W1 = data[0   : n_1]
        np.reshape(self.W1, (self.n_x, self.n_h1))
        self.W2 = data[n_1 : n_2]
        np.reshape(self.W2, (self.n_h1, self.n_h2))
        self.W3 = data[n_2 : n_3]
        np.reshape(self.W3, (self.n_h2, self.n_y))

        return self

    def forward(self, new_state, first_move=False):
        #self.state = new_state # COMMENTED

        # Service during the first move
        if first_move:
            #print("First move")
            return 1
        
        # Inputs vector multiplication by the first weights matrix
        h1 = np.maximum(0, new_state @ self.W1 ) # ReLU # CHANGE self.state -> new_state
        # First hidden layer matrix multiplication by the second weights matrix
        h2 = np.maximum(0, h1 @ self.W2) # ReLU

        # Use softmax as it is best for multivalue outputs 
        x = h2 @ self.W3
        e_x = np.exp(x - np.max(x))
        softmax = e_x / e_x.sum()

        outcome = int(np.argmax(softmax))
        outcome += 1 # Omit NOOP
       
        return outcome #int(np.argmax(softmax))
       
       #return int(np.argmax(softmax))

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