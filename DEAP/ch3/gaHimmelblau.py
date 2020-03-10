from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import elitism
import niching 
#Problem constants

DIMENTIONS = 2
BOUND_LOW, BOUND_UP = -5,5

# GA constants

POPULATION_SIZE = 300
P_CROSSOVER = 0.9 # prob of crossover
P_MUTATION = 0.5 # prob of mutation
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0
DISTANCE_THRESHOLD = 0.1
SHARING_EXTENT = 5.0
# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

def randomFloat(low,up,dimention):
    return [random.uniform(l,u) for l, u in zip([low]* dimention, [up]*dimention)]

toolbox.register("floatAttr", randomFloat,BOUND_LOW, BOUND_UP, DIMENTIONS)

# define a single objective, minimizing fitness strategy
creator.create("FitnessMin", base.Fitness, weights=(1.0,))

# create the Individual class based on list
creator.create("Individual",list, fitness=creator.FitnessMin)

# create the individual operator to fill up an Individual instance
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.floatAttr)

# create the population operator to generate a list of individuals
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

def himmelblau(individual):
    x = individual[0]
    y = individual[1]
    f = -(x**2+y-11)**2 - (x+y**2-7)**2
    return f+2000, #Move fitness values up by 2000
toolbox.register("evaluate", himmelblau)

# genetic operators:
toolbox.register("select", niching.selTournamentWithSharing, tournsize=2, distanceThreshold=DISTANCE_THRESHOLD,sharingExtent=SHARING_EXTENT)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR, indpb=1.0/DIMENTIONS)

# Genetic Algorithm
def main():

    # create initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
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
    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i].fitness.values[0], " -> ", hof.items[i])
    
    plt.figure(1)
    globalMaxima = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
    plt.scatter(*zip(*globalMaxima), marker='x', color='red', zorder=1)
    plt.scatter(*zip(*population), marker='.', color='blue', zorder=0)    # plot solution locations

    # plot best solutions locations 
    plt.figure(2)
    plt.scatter(*zip(*globalMaxima), marker='x', color='red', zorder=1)
    plt.scatter(*zip(*hof.items), marker='.', color='blue', zorder=0)

    # get statistics
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics
    plt.figure(3)
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')

    plt.show()



if __name__ == "__main__":
    main()