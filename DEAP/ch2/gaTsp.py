from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import array

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tsp

# problem constants:
# Using the problem class defined in tsp.py
TSP_NAME = "bayg29"
tsp = tsp.TravelingSalesmanProblem(TSP_NAME)

# Algortihm Constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.9 # prob of crossover
P_MUTATION = 0.1 # prob of mutation
MAX_GENERATIONS = 500
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

# Genetic operators 

toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(tsp))

# Genetic Algorithm flow:
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
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best individual info
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # plot best solution
    plt.figure(1)
    tsp.plotData(best)

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


if __name__ == "__main__":
    main()