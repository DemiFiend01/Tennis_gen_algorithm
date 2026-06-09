import pygad
import pygad.gann
import FixedAtariWrapper as faw
import ale_py
import gymnasium as gym
from individual import Individual
import numpy as np
import copy
import time
import random
from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject

# Fitness function notes:
# Statistics to track:
# - Number of ball hits

# - Won / lost

# 0. Perform 10 
#    * Main loop
#      - Make prediction of the next move
#      - Extract current ram
#      - Compare with previous
#      - Evaluate statistics
#      - After each game evaluate intermediary fitness
# 1. Measures
#    - If ball passes players x and delta_y is high -> big penalty
#    - Count number of consecutive hits -> means that agent is winning a service each time
#    - Track movements of a player -> penalty for sitting still
#    - When ball is travelling towards our site -> check rate of distance change between player & ball, the higher, the better
#    - Ratio points_scored / lost_points -> penalize low values

# max fitness: 1.0
# Conditions:
# - (ok) Distance between player and ball when lost
# - Direction of player movement when lost
# - Number of consecutive hits (the higher the better)
# - Position distribution on x, y axes (the flatter, the better)
# - When ball travelling to our site, differential (reate of change) of distance change (negated), the higher the better
# - (ok) Ratio points_scored / points_lost -> the higher the better

# Old fitness function:
'''
total_inv_reward = 0.0
prev_ball_x, prev_ball_y = 0,0

for i in range(1000):
    action = individual.forward(new_state=self.game_ram_state)
    if i % 30 == 0:
        action = 1
    obs, reward, terminated, truncated, info = self.env.step(action)

    labels = info["labels"]
    # print(labels)
    # print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {labels}")
    # print(self.game_ram_state)

    self.game_ram_state = self.extract_game_info(labels=labels, 
                                                    prev_ball_x=prev_ball_x, 
                                                    prev_ball_y=prev_ball_y)
    prev_ball_x = labels["ball_x"]
    prev_ball_y = labels["ball_y"]

    distance = abs(labels["player_x"] - labels["ball_x"]) + abs(labels["player_y"] - labels["ball_y"])
    proximity_reward = -distance / 255.0

    total_inv_reward += reward +  20 * proximity_reward
    if terminated or truncated:
        print("terminated or truncated")
        break

return total_inv_reward
'''


'''To Do:
- Add Selection methods:
- Choose one winner and select the best state.
- include a loop from pygad
'''

class GeneticAlgorithm:

    PL_RAM_IDX = 69
    EN_RAM_IDX = 70
    MAX_BALL_PLAYER_DIST = 200 # 199 actually but shhhhh * ~ - 

    def __init__(self, no_inv: int, no_gen, max_fit : float = 15.0, threads : int = 1):
        """Initialises the genetic algorithm

        Args:
            no_inv  (int)   : Number of individuals
            no_gen  (int)   : Number of generations
            max_fit (float) : Max evaluation time for the fitness function. If exceeded fitness func is terminated early. 
            threads (int)   : Number of threads used to evaluate the fitness function.
        """
        self.no_inv = no_inv
        self.no_gen = no_gen
        self.last_fitness = 0 
        self.env = OCAtari("Tennis-v4", mode="ram", hud=True, render_mode=None) # faw.FixedAtariWrapper(gym.make('ALE/Tennis-v5', render_mode=None)) # # Tennis-v4
        
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)

        self.game_objs = dict()
        self.game_ram_state = self.extract_game_info(self.env)

        initial_population = [] # not pygad.nn unfortunately, but the Individual class, because of supervised learning
        for i in range(self.no_inv):
            individual = Individual(state=self.game_ram_state)
            initial_population.append(individual.get_weights())
        initial_population = np.array(initial_population)

        self.max_fitness_time = max_fit # Maximal evaluation time of a fitness function

        if threads < 0: threads = 1
        parents_mating = no_inv / 5
        if parents_mating <= 1 : parents_mating = 2

        self.ga_instance = pygad.GA(
            num_generations=self.no_gen,
            num_parents_mating=parents_mating, #4
            initial_population=initial_population,#here an array or something of the individuals as vectors
            fitness_func=self.fitness_func, 
            parent_selection_type="sss", # we can swap it by saying "custom" or something
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10, #10
            on_generation=self.on_generation,
            parallel_processing=threads
        )

    def run(self):
        self.ga_instance.run()
        self.env.close()

    # Not used, dunno how to
    def selection_function(self, fitness, num_parents, ga_instance):

        fitness = np.array(fitness)
        parents_indices = []

        for _ in range(num_parents):
            candidates = np.random.choice(len(fitness), 3, replace=False)
            best_candidate = candidates[np.argmax(fitness[candidates])]
            parents_indices.append(best_candidate)

        return np.array(parents_indices)

    def on_generation(self, ga_instance):
        # Save the info of the best individual and showcase them, even showing their result
        best_sol, best_fit, x = self.ga_instance.best_solution()
        print(f"Gen {self.ga_instance.generations_completed} -> best fitness {best_fit}")
        self.showcase_best(solution=best_sol)

    def extract_game_info(self, env):

        enemy_x, enemy_y = 0, 0
        player_x, player_y = 0, 0
        enemy_result, player_result = 0, 0
        ball_x, ball_y = 0, 0
        ball_vx, ball_vy = 0, 0
        ball_v = 0

        for obj in env.objects:
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

        RAM = env.get_ram()
        enemy_result = RAM[70]
        player_result = RAM[69]

        self.game_ram_state = np.array([
            enemy_x, enemy_y,
            player_x, player_y,
            enemy_result, player_result,
            ball_x, ball_y,
            ball_vx, ball_vy,
            ball_v
        ], dtype=np.float32)

        # Normalizing the value so that it does not drift too high up later
        # self.game_ram_state /= 255.0

        return self.game_ram_state

    def get_ram_objects(self, env):
        for obj in env.objects:
                self.game_objs[obj.category.lower()] = obj
        return self.game_objs

    def fitness_func(self, ga_instance, solution, sol_idx):
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(1)

        individual = Individual(self.extract_game_info(self.env))
        individual.set_weights(solution)

        service_now = True 

        deadlock_counter = 0

        def detect_deadlock(env, objs):
            ''' It's supposed to counter error when player runs of in random direction before serving the ball. '''
            nonlocal deadlock_counter

            if "ball" in objs.keys():
                ball = objs["ball"]
            else:
                print("No ball among objects, skipping")
                return False
            
            if "player" in objs.keys():
                player = objs["player"]
            else:
                print("No player detected")
                return False

            if ball.prev_xy[0] == ball.x and player.prev_xy == (player.x, player.y):
                deadlock_counter += 1

            if deadlock_counter == 100:
                print("Detected deadlock")
                if ball.x >= 0 and ball.y >= 0:
                    print("Setting ram")
                    env.set_ram(26, ball.x)
                    env.set_ram(24, ball.y)
                    deadlock_counter = 0
                    return True
            return False

        print("\n\nNew Individual\n\n")

        player_total_score    = 0
        enemy_total_score     = 0
        total_missed_distance = [1.0] # Total accumulated distance between player and the ball at the moment player lost it
        total_missed_cos      = [0.0] # Total cos of angle between player movement direction and the vector from player to the balls position
        previous_ram          = copy.deepcopy(self.env.get_ram())
        total_time = 0                # Runtime of the fitness function
        step_counter = 0

        while not (terminated or truncated):
            start_time = time.time()
            step_counter += 1

            if service_now:
                action = 1
                service_now = False
            else:
                action = individual.forward(new_state=self.game_ram_state)

            obs, reward, terminated, truncated, info = self.env.step(action)

            self.game_ram_state = self.extract_game_info(self.env)
            objs = self.get_ram_objects(self.env)

            RAM = self.env.get_ram()

            if RAM[self.PL_RAM_IDX] > previous_ram[self.PL_RAM_IDX] and service_now == False:
                service_now = True
                player_total_score += 1

            if RAM[self.EN_RAM_IDX] > previous_ram[self.EN_RAM_IDX] and service_now == False:
                service_now = True
                enemy_total_score += 1 

                player = objs["player"]
                ball   = objs["ball"]

                dx = ball.x - player.x
                dy = ball.y - player.y
                ball_player_dist = ((dx ** 2 + dy ** 2) ** 0.5) / self.MAX_BALL_PLAYER_DIST
                total_missed_distance.append(ball_player_dist)

                dx_p = player.x - player.prev_xy[0]
                dy_p = player.y - player.prev_xy[1]
                p = np.asarray([dx_p, dy_p])
                b = np.asarray([dx, dy])
                c = np.dot(p, b) / (np.linalg.norm(p) * np.linalg.norm(b)) if (np.all(b != 0) and np.all(p != 0)) else 0.0
                total_missed_cos.append(c)

                if detect_deadlock(self.env, objs):
                    service_now = True

            previous_ram = copy.deepcopy(self.env.get_ram()) 

            total_time += (time.time() - start_time)

            if total_time > self.max_fitness_time:
                print(f"Terminating early: individual exceeded time budget ({total_time:.2f}s)")
                break
                
            if step_counter > 5000:
                print("Terminating early: hit max safety step threshold (5000 frames)")
                break

        total_missed_distance = sum(total_missed_distance) / len(total_missed_distance)
        total_missed_cos = sum(total_missed_cos) / len(total_missed_cos)
        player_success_fraction = player_total_score / (player_total_score + enemy_total_score)

        print(f"total missed distance: {total_missed_distance}")
        print(f"total missed cos:      {total_missed_cos}")
        print(f"player success frac:   {player_success_fraction}")

        f = player_success_fraction + (1.0 - total_missed_distance) + total_missed_cos

        print(f"Fitness: {f}")
        return f #player_won_fraction # For now just return the percentage of won points

    def showcase_best(self, solution):
        '''
        env = OCAtari("Tennis-v4", mode="ram", hud=True, render_mode="human")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        individual = Individual(self.extract_game_info(env))
        individual.set_weights(solution)

        for i in range(500):
            action = individual.forward(new_state=self.game_ram_state)
            if i % 30 == 0:
                action = 1
            obs, reward, terminated, truncated, info = env.step(action)

            objs = self.get_ram_objects(env)
            if "ball" in objs.keys() and "player" in objs.keys():
                print(f"({objs['player'].x}, {objs['player'].y}), ({objs['ball'].x}, {objs['ball'].y})")
            #print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {labels}")
            #print(self.game_ram_state)

            self.game_ram_state = self.extract_game_info(env)
            if terminated or truncated:
                break
        '''

        return 1.0
