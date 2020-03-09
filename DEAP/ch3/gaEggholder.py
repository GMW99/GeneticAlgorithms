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

toolbox = base.Toolbox()

def randomFloat(low,up,dimention):
    return [random.uniform(1,u) for 1, u in zip([low]* dimention, [up]*dimention)]
toolbox.register("floatAttr", randomFloat,BOUND_LOW, BOUND_UP, DIMENTIONS)

# define a single objective, minimizing fitness strategy
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list
creator.create("Individual",list, fitness=creator.FitnessMin)

# create the individual operator to fill up an Individual instance
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.floatAttr)

# create the population operator to generate a list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def eggholder(individual):
    x = individual[0]
    y = individual[1]
    f = -(y+47)*np.sin(np.sqrt(np.abs(x/2+(y+47))))-x*np.sin(np.sqrt(np.abs(x-(y+47))))
    return f,
toolbox.register("evaluate", eggholder)