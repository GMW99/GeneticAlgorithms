from deap import base
from deap import creator 
from deap import tools


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import friedman
import elitism
import random
# Problem constants 

NUM_FEATURES = 15
NUM_SAMPLES = 60

# Genetic Algorithm constants
POPULATION_SIZE = 30 # number of individuals in population
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.3 # probability for mutating an individual
MAX_GENERATIONS = 30 # max  number of generations for stopping condition
HALL_OF_FAME_SIZE = 5

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

# Main GA loop

def main():
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best individual info
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])


    # plot statistics
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red', label='minFitness')
    plt.plot(meanFitnessValues, color='green', label='avgFitness')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots
    plt.show()

if __name__  == "__main__":
    main()