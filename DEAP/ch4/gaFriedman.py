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
friedman = friedman.Friedman1(NUM_FEATURES,NUM_SAMPLES,RANDOM_SEED)

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint,0 ,1)
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator",tools.initRepeat,creator.Individual,toolbox.zeroOrOne, len(friedman))
toolbox.register("populationCreator",tools.initRepeat,list,toolbox.individualCreator)

def friedmanFitness(individual):
  return friedman.getMSE(individual),

toolbox.register("evaluate", friedmanFitness)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(friedman))
