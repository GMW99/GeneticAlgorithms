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
P_CROSSOVER = 0.8 # prob of crossover
P_MUTATION = 0.15 # prob of mutation
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 10
CROWDING_FACTOR = 20.0

# Setting the seed so we get the same result each time
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

def randomFloat(low,up,dimention):
    return [random.uniform(l,u) for l, u in zip([low]* dimention, [up]*dimention)]
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

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENTIONS)

# Genetic Algorithm
def main():

    # create initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # define the hall-of-fame object
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    # get statistics
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.show()


if __name__ == "__main__":
    main()