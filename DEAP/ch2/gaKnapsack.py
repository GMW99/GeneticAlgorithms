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

# toolbox set up

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calcuation
def knapsackValue(individual):
    return knapsack.getValue(individual),  # return a tuple

toolbox.register("evaluate", knapsackValue)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack))
