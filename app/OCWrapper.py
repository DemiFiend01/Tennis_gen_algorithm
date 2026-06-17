from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject
import numpy as np
import math
from typing import Dict, List, Tuple

class OCWrapper(OCAtari):
    '''
    Wraps the OCAtari environment. 

    Fields:
        env (OCAtari)          : Local environment.
        deadlock_counter (int) : Used to count frames in self.detect_deadlock().
    '''

    BALL_RAM_X = 16
    BALL_RAM_Y = 17
    PLAYER_RAM_X = 26
    PLAYER_RAM_Y = 25 # 24 in atariari is wrong
    PLAYER_RAM_SCORE = 69  # IDX
    ENENEMY_RAM_SCORE = 70 # IDX
    COURT_WIDTH = 125  # 141 - 16
    COURT_HEIGHT = 135 # 159 - 24
    MAX_BALL_PLAYER_DIST = 200 # 199 actually but shhhhh * ~ - 
    MAX_DIST = 180

    # Player x-axis bounds for player to be in the deadlock
    DEADLOCK_BOUND_LEFT  = 30
    DEADLOCK_BOUND_RIGHT = 130
    DEADLOCK_THRESHOLD   = 30  # Number of consecutive frames counted for deadlock to happen

    # Zone where ball is stuck if service fails
    BALL_SERVICE_ZONE_X = 75
    BALL_SERVICE_ZONE_Y_MIN = 20
    BALL_SERVICE_ZONE_Y_MAX = 50

    def __init__(self, type : str, mode : str, hud : bool, render_mode : str=None):
        '''
        Initializes the wrapper. 
        See https://oc-atari.readthedocs.io/en/latest/ocatari/core.html for details.
        '''
        super().__init__(env_name=type, mode=mode, hud=hud, render_mode=render_mode)
        #self.env = OCAtari(type, mode=mode, hud=hud, render_mode=render_mode)
        self.deadlock_counter = 0

    def get_ram_objects(self) -> Dict[str, GameObject]:
        ''' 
        Returns the dictionary of game objects from the self.objects list. 
        Lowercase object category name is the key and object itself is the value.
        '''
        game_objs = {}
        for obj in self.objects:
                game_objs[obj.category.lower()] = obj
        return game_objs

    def extract_game_info(self):
        '''
        Extracts game data from the OCAtari env & from the game RAM.
        Returns (np.array) : [
            enemy_x, enemy_y, player_x, player_y,
            enemy_result, player_result,
            ball_x, ball_y, ball_vx, ball_vy, ball_v
            ] as np.float32.
        '''

        enemy_x, enemy_y = 0, 0
        player_x, player_y = 0, 0
        enemy_result, player_result = 0, 0
        ball_x, ball_y = 0, 0
        ball_vx, ball_vy = 0, 0
        ball_v = 0

        for obj in self.objects:
            if obj.category == "Player":
                player_x = obj.x
                player_y = obj.y
            elif obj.category == "Enemy":
                enemy_x = obj.x
                enemy_y = obj.y
            elif obj.category == "Ball":
                ball_x = obj.x
                ball_y = obj.y
                ball_vx = obj.x - obj.prev_xy[0]
                ball_vy = obj.y - obj.prev_xy[1]
                ball_v = np.sqrt(ball_vx ** 2 + ball_vy ** 2)

        RAM = self.get_ram()
        enemy_result = RAM[70]
        player_result = RAM[69]

        # Used to be normalized in version where RAM values were directly read
        return np.array([
            enemy_x, enemy_y,
            player_x, player_y,
            enemy_result, player_result,
            ball_x, ball_y,
            ball_vx, ball_vy,
            ball_v
        ], dtype=np.float32)

    def detect_deadlock(self):
        ''' 
        Detects situation in which ball.x == const && player.xy == const. 
        Counts number of frames spent in such state.
        If number of frames execeeds threshold, deadlock is assumed.

        Returns:
            (bool) : True if deadlock has been detected, false otherwise.
        '''
        objs = self.get_ram_objects()

        # 1. Check if player and ball objects are detected

        if "ball" in objs.keys():
            ball = objs["ball"]
        else:
            #print("No ball among objects, skipping")
            return False
        if "player" in objs.keys():
            player = objs["player"]
        else:
            return False
        
        # 2. If ball.x == const & players position == const, count that as deadlock

        #if ((ball.prev_xy[0] == ball.x and player.prev_xy == (player.x, player.y)) and (player.x < self.DEADLOCK_BOUND_LEFT or player.x > self.DEADLOCK_BOUND_RIGHT)) or :
        if ball.x == self.BALL_SERVICE_ZONE_X and ball.y >= self.BALL_SERVICE_ZONE_Y_MIN and ball.y <= self.BALL_SERVICE_ZONE_Y_MAX :

            #print(f"Possible deadlock: {ball.prev_xy}, {player.prev_xy}")
            self.deadlock_counter += 1
        else:
            self.deadlock_counter = 0

        # 3. In case of deadlock teleport player to the balls (x, y) coordinates

        if self.deadlock_counter == self.DEADLOCK_THRESHOLD:
            #print("Detected deadlock")
            if ball.x >= 0 and ball.y >= 0:
                #print("Setting ram")

                self.set_ram(self.PLAYER_RAM_X, ball.x)
                self.set_ram(self.PLAYER_RAM_Y, ball.y) 
                self.deadlock_counter = 0
                return True
            
        return False

    def ball_started_moving(self):
        ''' Returns True if ball has started to move after a succesfull service. '''
        objs = self.get_ram_objects()
        if 'ball' in objs.keys():
            ball = objs['ball']
            dx = ball.prev_xy[0] - ball.x
            
            return dx != 0 and ball.y > self.BALL_SERVICE_ZONE_Y_MAX
        return False

    def obj_dist(self, o1, o2):
            return math.sqrt((o1.x - o2.x) ** 2 + (o1.y - o2.y) ** 2)

