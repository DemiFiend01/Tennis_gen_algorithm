import pygad
import pygad.gann
import FixedAtariWrapper as faw
import ale_py
import gymnasium as gym
from individual import Individual
import numpy as np
import copy, time, random, math, re
from ocatari.core import OCAtari
from ocatari.ram.game_objects import GameObject


from OCWrapper import OCWrapper
from pathlib import Path

INDIVIDUALS_PATH = Path("/tmp/best_fits")

# Fitness function notes:
# Statistics to track:
# - Number of ball hits

# - Won / lost

#-1. (ok) Create env in the 
# 0. (ok) Fewer outputs
# 1. (ok) Less randomness in actions
# 2. (ok) Calculate distance between player and ball always when ball is on player side
# 3. Successfull hits -> every time balls movement direction inverted when near player
# 4. (ok) Number of survived frames -> modify to count frames per game 

# 5. Modify the fitenss function ->
#    - Add flags to GeneticAlgorithm
#       - Each flag is unlocked after achieving some fitness function threshold or number of generations 
#       - Flags: ball_player_dist, total_survived_frames, total_number_of_hits, 
#    - Activate new elements of fitness function gradually
#   - Each new metric unlocked after 8-10 generations

class GeneticAlgorithm:
    NN_X = 11 # Number of inputs for single individual
    NN_Y = 10 # Number of outputs of a single individual

    def __init__(self, no_inv : int,
                       no_gen : int,
                       parents_frac  : float = 0.3,
                       max_fit_time  : float = 15.0, 
                       max_fit_steps : int = 5000,
                       threads : int = 1):
        """
            Initialises the genetic algorithm

        Args:
            no_inv (int)         : Number of individuals.
            no_gen (int)         : Number of generations.
            parents_frac (float) : Fraction of all individuals chosen for mating. 
            max_fit_steps (int)  : Max number of steps made in a fitness function.
            max_fit_time (float) : Max evaluation time for the fitness function. If exceeded fitness func is terminated early. 
            threads (int)        : Number of threads used to evaluate the fitness function.
        """
        self.no_inv = no_inv
        self.no_gen = no_gen
        initial_population = self.init_population(no_inv)

        self.max_steps = max_fit_steps
        self.max_fitness_time = max_fit_time

        if threads < 0: threads = 1
        
        parents_mating = int(no_inv * parents_frac)
        if parents_mating <= 1 : parents_mating = 2

        self.ga_instance = pygad.GA(
            num_generations=self.no_gen,
            num_parents_mating=parents_mating,     
            initial_population=initial_population, # here an array or something of the individuals as vectors
            fitness_func=self.fitness_func, 
            parent_selection_type="sss",           # we can swap it by saying "custom" or something
            keep_parents=1,
            crossover_type="uniform",
            mutation_type="random",              # Gaussian would be nbetter for Small shifts to existing weights instead of pure randomness
            random_mutation_min_val=-0.15,
            random_mutation_max_val=0.15,
            mutation_by_replacement=False,
            mutation_percent_genes=3,              # We don't want to alter the nn-s too much
            on_generation=self.on_generation,
            parallel_processing=threads
        )
    
    def init_population(self, no_inv):
        '''
            Creates the initial population based on self.NN_X, self.NN_Y.
        '''
        '''
        local_env = OCWrapper(type="Tennis-v4", mode="ram", hud=True, render_mode=None)
        local_env.reset()
        local_env.step(0)
        game_ram_state = self.extract_game_info(local_env)

        initial_population = [] 
        for i in range(self.no_inv):
            individual = Individual(inputs_num=len(game_ram_state), n_y=10) 
            initial_population.append(individual.get_weights())
        return np.array(initial_population)
        '''
        initial_population = [] 
        for i in range(self.no_inv):
            individual = Individual(inputs_num=self.NN_X, n_y=self.NN_Y) 
            initial_population.append(individual.get_weights())
        return np.array(initial_population)

    def run(self):
        self.ga_instance.run()

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

        # Save best indivdual into a file
        current_gen = ga_instance.generations_completed
        if current_gen % 2 == 0:
            INDIVIDUALS_PATH.mkdir(parents=True, exist_ok=True)
            solution, solution_fitness, _ = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
            solution_fitness = round(solution_fitness, 4)
            np.save(INDIVIDUALS_PATH / f"gen-{current_gen}-fit-{solution_fitness}", arr=solution)

    def fitness_func(self, ga_instance, solution, sol_idx):
        # 0. Prepare local env and the individual to evaluate

        local_env = OCWrapper(type="Tennis-v4", mode="ram", hud=True, render_mode=None) # render_mode=None "human"
        obs, info = local_env.reset()
        obs, reward, terminated, truncated, info = local_env.step(1)

        individual = Individual(inputs_num=self.NN_X, n_y=self.NN_Y)
        individual.set_weights(solution)

        print("\n\nNew Individual\n\n")

        # 1. Evaluation metrics

        player_total_score    = 0 
        enemy_total_score     = 0
        total_missed_distance = []      # Average distance between player and ball at the moment player loses ability to succefully return it to the opponent
        avg_player_ball_dist  = []      # Average distance between the player and the ball through out the whole game
        move_it_move_it = 0.0
        player_prev_pos = [ 0, 0]

        # total_missed_cos      = []      # Total cos of angle between player movement direction and the vector from player to the balls position
        previous_ram          = copy.deepcopy(local_env.get_ram())
        total_time   = 0                # Runtime of the fitness function
        step_counter = 0

        service_now = True              # Should the NN make service unconditionally now?
        start_time = time.time()

        while not (terminated or truncated):
            step_counter += 1

            local_game_ram_state = local_env.extract_game_info()
            objs = local_env.get_ram_objects()
            RAM = local_env.get_ram()

            action = individual.forward(new_state=local_game_ram_state, first_move=service_now)
            obs, reward, terminated, truncated, info = local_env.step(action)

            #-1. Detect successfull service
            if service_now:
                if local_env.ball_started_moving():
                    service_now = False

            # -------------------- Metrics calculation -------------------- #

            
            # 0. If ball is flying towards the player, check if the player is closing in on the ball
            if 'ball' in objs.keys():
                ball   = objs['ball']
                player = objs['player']
                ball_vy = ball.y - ball.prev_xy[1]
                if ball_vy > 0:
                    dist = local_env.obj_dist(ball, player) / local_env.MAX_DIST # (objs['ball'], objs['player']) / self.MAX_DIST
                    avg_player_ball_dist.append(dist)

            # 1. Check if players score changed
            if RAM[local_env.PLAYER_RAM_SCORE] > previous_ram[local_env.PLAYER_RAM_SCORE] and service_now == False:
                service_now = True
                player_total_score += 1

            # 2. Check if player should be serving in the next round
            # 3. Check player movement direction
            if RAM[local_env.ENENEMY_RAM_SCORE] > previous_ram[local_env.ENENEMY_RAM_SCORE] and service_now == False:
                service_now = True
                enemy_total_score += 1 

                '''
                player = objs["player"]
                ball   = objs["ball"]

                dx = ball.x - player.x
                dy = ball.y - player.y
                ball_player_dist = ((dx ** 2 + dy ** 2) ** 0.5) / local_env.MAX_BALL_PLAYER_DIST
                total_missed_distance.append(ball_player_dist)

                dx_p = player.x - player.prev_xy[0]
                dy_p = player.y - player.prev_xy[1]
                p = np.asarray([dx_p, dy_p])
                b = np.asarray([dx, dy])
                c = np.dot(p, b) / (np.linalg.norm(p) * np.linalg.norm(b)) if (np.all(b != 0) and np.all(p != 0)) else 0.0
                total_missed_cos.append(c)
                '''
            player_temp = objs['player']
            if not service_now:
                if not abs(player_temp.x - player_prev_pos[0]) and not abs(player_temp.y - player_prev_pos[1]):
                    move_it_move_it += -0.1
                else:
                    move_it_move_it += 0.2
            player_prev_pos = [player_temp.x, player_temp.y]

            # 4. Detect deadlock
            if local_env.detect_deadlock():
                service_now = True

            previous_ram = copy.deepcopy(local_env.get_ram()) 
            total_time = time.time() - start_time

            # 5. Count steps & measure total exec time
            if total_time >= self.max_fitness_time:
                print(f"Terminating early: individual exceeded time budget ({total_time:.2f}s)")
                break
            if step_counter >= self.max_steps:
                print(f"Terminating early: hit max safety step threshold ({self.max_steps} steps)")
                break

        # 6. Normalize metrics
        if step_counter != 0:
            move_it_move_it = move_it_move_it / step_counter

        if len(total_missed_distance) != 0:
            total_missed_distance = sum(total_missed_distance) / len(total_missed_distance)
        else:
            total_missed_distance = 0

        if len(avg_player_ball_dist) != 0:
            avg_player_ball_dist = sum(avg_player_ball_dist) / len(avg_player_ball_dist)
        else:
            avg_player_ball_dist = 0

        '''
        if len(total_missed_cos) != 0:
            total_missed_cos = sum(total_missed_cos) / len(total_missed_cos)
        else:
            total_missed_cos = 0
        '''

        if player_total_score != 0 or enemy_total_score !=0:
            player_success_fraction = player_total_score / (player_total_score + enemy_total_score)
        else:
            player_success_fraction = 0

        # 7. Omit deadlocks & loops
        if step_counter < self.max_steps or player_total_score > 0:
            total_frames_survived = step_counter / self.max_steps
        else: 
            total_frames_survived = 0

    #    print(f"total missed distance:  {total_missed_distance}")
    #    print(f"avg player ball dist:   {avg_player_ball_dist}")
    #    print(f"total missed cos:       {total_missed_cos}")
    #    print(f"player success frac:    {player_success_fraction}")
    #    print(f"totatl survived frames: {total_frames_survived}")

        # 8. Calculate fitness

        current_gen = ga_instance.generations_completed

        f = (1.0 - avg_player_ball_dist) # + total_frames_survived + (1.0 - total_missed_distance) + player_success_fraction # total_missed_cos 
        f += move_it_move_it
        print(f"Move it: {move_it_move_it}")
        if current_gen > 10:
            f += total_frames_survived
        if current_gen > 20:
            f += 1.0 - total_missed_distance
        if current_gen > 30:
            f += player_success_fraction
        
        print(f"Fitness: {f}")
        return f 

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
