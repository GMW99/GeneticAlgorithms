from deap import base
from deap import creator 
from deap import tools

import random
import matplotlib.pyplot as plt

import friedman
import elitism

# Problem constants 

NUM_FEATURES = 15
NUM_SAMPLES = 60

# Genetic Algorithm constants
POPULATION_SIZE = 30 # number of individuals in population
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.1 # probability for mutating an individual
MAX_GENERATIONS = 30 # max  number of generations for stopping condition
HALL_OF_FRAME_SIZE = 3

# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)