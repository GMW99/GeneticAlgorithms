from deap import base
from deap import creator
from deap import tools
from deap import algortihms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import knapsack

# problem constants:
# Using the problem class defined in knapsack.py

knapsack= knapsack.Knapsack01Problem()

# Algortihm Constants:

POPULATION_SIZE = 50
P_CROSSOVER = 0.9 # prob of crossover
P_MUTATATION = 0.1 # prob of mutation
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1

# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
