from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import tsp

# problem constants:
# Using the problem class defined in tsp.py
TSP_NAME = "bayg29"
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)

# Algortihm Constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9 # prob of crossover
P_MUTATATION = 0.1 # prob of mutation
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 1

# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# toolbox set up

toolbox = base.Toolbox()

# create an operator that randomly shuffles indices
toolbox.register("randomOrder", random.sample, range(len(tsp)), len(tsp))

# define a single objective, minimizing fitness strategy
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# create the individual operator to fill up an Individual instance
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population operator to generate a list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculator, calcuates the total distance of the list represented
def tspDistance(individual):
    return tsp.getTotalDistance(individual), # return tuple

toolbox.register("evaluate", tspDistance)