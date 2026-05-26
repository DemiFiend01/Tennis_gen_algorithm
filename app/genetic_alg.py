import pygad
import pygad.gann
import FixedAtariWrapper as faw
import ale_py
import gymnasium as gym
from individual import Individual
import numpy as np
'''To Do:
- Add Selection methods:
- Choose one winner and select the best state.
- include a loop from pygad
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
        self.last_fitness = 0
        self.env = faw.FixedAtariWrapper(gym.make('ALE/Tennis-v5', render_mode=None))
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)

        self.game_ram_state = self.extract_game_info(info["labels"])

        initial_population = []# not pygad.nn unfortunately, but the Individual class, because ot supervised learning
        for i in range(self.no_inv):
            individual = Individual(state=self.game_ram_state)
            initial_population.append(individual.get_weights())
        initial_population = np.array(initial_population)
        
        self.ga_instance = pygad.GA(
            num_generations=self.no_gen,
            num_parents_mating=4, #4
            initial_population=initial_population,#here an array or something of the individuals as vectors
            fitness_func=self.fitness_func,
            parent_selection_type="sss", # we can swap it by saying "custom" or something
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10, #10
            on_generation=self.on_generation
        )

    def run(self):
        self.ga_instance.run()
        self.env.close()
        
    def on_generation(self, ga_instance):
        # Save the info of the best individual and showcase them, even showing their result
        best_sol, best_fit, x = self.ga_instance.best_solution()
        print(f"Gen {self.ga_instance.generations_completed} -> best fitness {best_fit}")
        self.showcase_best(solution=best_sol)

    def extract_game_info(self, labels, prev_ball_x = 0, prev_ball_y = 0):
        enemy_x = labels['enemy_x']
        enemy_y = labels['enemy_y']
        player_x = labels['player_x']
        player_y = labels['player_y']
        enemy_result = labels['enemy_score']
        player_result = labels['player_score']
        ball_x = labels['ball_x']
        ball_y = labels['ball_y']

        #Calculated values
        ball_dir_x = float(ball_x) - float(prev_ball_x)
        ball_dir_y = float(ball_y) - float(prev_ball_y)
        ball_v = np.sqrt(ball_dir_x**2 + ball_dir_y**2)

        self.game_ram_state = np.array([
            enemy_x, enemy_y,
            player_x, player_y,
            enemy_result, player_result,
            ball_x, ball_y,
            ball_dir_x, ball_dir_y,
            ball_v
        ], dtype=np.float32)
        #print(self.game_ram_state)

        # Normalizing the value so that it does not drift too high up later
        self.game_ram_state /= 255.0

        return self.game_ram_state

    def fitness_func(self, ga_instance, solution, sol_idx):
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)

        individual = Individual(self.extract_game_info(info["labels"]))
        individual.set_weights(solution)

        total_inv_reward = 0.0
        prev_ball_x, prev_ball_y = 0,0

        for i in range(500):
            action = individual.forward(new_state=self.game_ram_state)
            if i % 30 == 0:
                action = 1
            obs, reward, terminated, truncated, info = self.env.step(action)

            labels = info["labels"]
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
                break

        return total_inv_reward

    def showcase_best(self, solution):
        env = faw.FixedAtariWrapper(gym.make('ALE/Tennis-v5', render_mode="human"))
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        individual = Individual(self.extract_game_info(info["labels"]))
        individual.set_weights(solution)

        total_inv_reward = 0.0
        prev_ball_x, prev_ball_y = 0,0

        for i in range(500):
            action = individual.forward(new_state=self.game_ram_state)
            obs, reward, terminated, truncated, info = env.step(action)

            labels = info["labels"]
            p_x = labels['player_x']
            b_x = labels['ball_x']
            print(f"Player x: {p_x}, ball_x: {b_x}, diff: {abs(p_x-b_x)}")
            #print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {labels}")
            #print(self.game_ram_state)

            self.game_ram_state = self.extract_game_info(labels=labels, 
                                                         prev_ball_x=prev_ball_x, 
                                                         prev_ball_y=prev_ball_y)
            prev_ball_x = labels["ball_x"]
            prev_ball_y = labels["ball_y"]
            if terminated or truncated:
                break

        return total_inv_reward
