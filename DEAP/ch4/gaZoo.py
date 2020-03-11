from deap import base
from deap import creator 
from deap import tools


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import zoo
import elitism
import random



# Genetic Algorithm constants
POPULATION_SIZE = 50 # number of individuals in population
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.2 # probability for mutating an individual
MAX_GENERATIONS = 50 # max  number of generations for stopping condition
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001

# Setting the seed so we get the same result each time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
zoo = zoo.Zoo(RANDOM_SEED)

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint,0 ,1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator",tools.initRepeat,creator.Individual,toolbox.zeroOrOne, len(zoo))
toolbox.register("populationCreator",tools.initRepeat,list,toolbox.individualCreator)

def zooFitness(individual, featurePenaltyFactor):
    numFeaturesUsed = sum(individual)
    if numFeaturesUsed == 0:
        return 0.0,
    else:
        accuracy = zoo.getMeanAccuracy(individual)
        return accuracy - featurePenaltyFactor * numFeaturesUsed,
        

toolbox.register("evaluate", zooFitness,\
                featurePenaltyFactor=FEATURE_PENALTY_FACTOR)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(zoo))

# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best solution found:
    print("- Best solutions are:")
    for i in range(HALL_OF_FAME_SIZE):
        print(i, ": ", hof.items[i], ", fitness = ", hof.items[i].fitness.values[0],
              ", accuracy = ", zoo.getMeanAccuracy(hof.items[i]), ", features = ", sum(hof.items[i]))

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()