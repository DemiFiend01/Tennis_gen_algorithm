from OCWrapper import OCWrapper
from individual import Individual
from pathlib import Path
import copy, re

class TestEnv():
    ''' Creates a test environment for the best fit individual and runs it in human readable mode. '''

    NN_X = 11 # Number of inputs for single individual
    NN_Y = 10 # Number of outputs of a single individual

    def __init__(self, dir : Path | str):

        if type(dir) is not Path:
            file = Path(dir)

        fittest_path = self._find_fittest(file)

        self.nn = Individual(inputs_num=self.NN_X, n_y=self.NN_Y).from_npy(fittest_path)
        self.env = OCWrapper(type="Tennis-v4", mode="ram", hud=True, render_mode="human")

    def _find_fittest(self, dir : Path):
        ''' Finds the fittest individual from all saved individuals. '''
        best_path = Path("")
        best_fitenss = 0.0
        pattern = r"(?:fit-(\d+\.*\d*))"
        for p in dir.resolve().iterdir():
            if m := re.search(pattern, str(p)):
                fitness = float(m.group(1))
                if fitness > best_fitenss:
                    best_fitenss = fitness
                    best_path = p.resolve()

        return best_path

    def start(self):
        self.env.reset()
        
        ram = self.env.get_ram()
        prev_ram = self.env.get_ram()
        first_move = True
        try:
            while True:
                game_state = self.env.extract_game_info()
                ram = self.env.get_ram()
                step = self.nn.forward(new_state=game_state, first_move=first_move)
                self.env.step(step)
                if first_move and self.env.ball_started_moving():
                    first_move = False
                first_move = not self.score_changed(ram, prev_ram, first_move)
                prev_ram = copy.deepcopy(ram) #copy.deepcopy(self.env.get_ram()) 
                if self.env.detect_deadlock(): first_move = True
        except KeyboardInterrupt:
            print("Exiting testing env.")
        finally:
            self.env.close()
    
    def score_changed(self, ram, prev_ram, first_move):
        if first_move:
            return False
        
        if ((ram[self.env.PLAYER_RAM_SCORE] != prev_ram[self.env.PLAYER_RAM_SCORE]) or 
            (ram[self.env.ENENEMY_RAM_SCORE] != prev_ram[self.env.ENENEMY_RAM_SCORE])
        ):
            return True
        
        return False