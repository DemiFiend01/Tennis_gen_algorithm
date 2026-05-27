import numpy
import pygad
import pygad.nn
import pygad.gann

import pygame 
import gymnasium as gym
import FixedAtariWrapper as faw

# Contains the basic tutorial from pygad.
# The outline of operation will be probably similiar but with modifications.

class Tutorial:
    def __init__(self, env : faw.AtariARIWrapper):

        # Envrionment wrapper
        self.env = env

        # Holds the fitness value of the previous generation.
        self.last_fitness = 0

        # Preparing the NumPy array of the inputs.
        #self.data_inputs = numpy.array([[2, 5, -3, 0.1],
         #                       [8, 15, 20, 13]])

        # Preparing the NumPy array of the outputs.
        #self.data_outputs = numpy.array([[0.1, 0.2],
         #                           [1.8, 1.5]])

        # The length of the input vector for each sample (i.e. number of neurons in the input layer).
        self.num_inputs = len(self.env.get_labels())

        # Creating an initial population of neural networks. The return of the initial_population() function holds references to the networks, not their weights. Using such references, the weights of all networks can be fetched.
        num_solutions = 6 # A solution or a network can be used interchangeably.
        self.GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                        num_neurons_input=num_inputs,
                                        num_neurons_hidden_layers=[2],
                                        num_neurons_output=2,
                                        hidden_activations=["relu"],
                                        output_activation="None")
        
        # population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
        # If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
        population_vectors = pygad.gann.population_as_vectors(population_networks=self.GANN_instance.population_networks)

        # To prepare the initial population, there are 2 ways:
        # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
        # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
        initial_population = population_vectors.copy()

        num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.

        num_generations = 500 # Number of generations.

        mutation_percent_genes = 5 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

        parent_selection_type = "sss" # Type of parent selection.

        crossover_type = "single_point" # Type of the crossover operator.

        mutation_type = "random" # Type of the mutation operator.

        keep_parents = 1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

        init_range_low = -1
        init_range_high = 1

        self.ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            initial_population=initial_population,
                            fitness_func=self.fitness_func,
                            mutation_percent_genes=mutation_percent_genes,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            parent_selection_type=parent_selection_type,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            keep_parents=keep_parents,
                            on_generation=self.callback_generation)

    def fitness_func(self, ga_instance, solution, sol_idx):
        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[sol_idx],
                                    data_inputs=self.data_inputs, problem_type="regression")
        solution_fitness = 1.0/numpy.mean(numpy.abs(predictions - self.data_outputs))

        return solution_fitness

    def callback_generation(self, ga_instance):
        population_matrices = pygad.gann.population_as_matrices(population_networks=self.GANN_instance.population_networks,
                                                                population_vectors=ga_instance.population)

        self.GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

        print(f"Generation = {ga_instance.generations_completed}")
        print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
        print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness}")

        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1].copy()

    def run(self):
        
        self.ga_instance.run()

        # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
        self.ga_instance.plot_fitness()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution(pop_fitness=self.ga_instance.last_generation_fitness)
        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")

        if self.ga_instance.best_solution_generation != -1:
            print(f"Best fitness value reached after {self.ga_instance.best_solution_generation} generations.")

        # Predicting the outputs of the data using the best solution.
        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[solution_idx],
                                    data_inputs=self.data_inputs,
                                    problem_type="regression")
        print(f"Predictions of the trained network : {predictions}")

        # Calculating some statistics
        abs_error = numpy.mean(numpy.abs(predictions - self.data_outputs))
        print(f"Absolute error : {abs_error}.")