from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import elitism

#Problem constants

DIMENTIONS = 2
BOUND_LOW, BOUND_UP = -512,512

# GA constants

POPULATION_SIZE = 300
P_CROSSOVER = 0.9 # prob of crossover
P_MUTATION = 0.1 # prob of mutation
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0

# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)