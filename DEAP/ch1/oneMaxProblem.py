from deap import base
from deap import creator 
from deap import tools

import random
import matplotlib.pyplot as plt
# Problem constants
ONE_MAX_LENGTH = 100

# Genetic Algorithm constants
POPULATION_SIZE = 200 # number of individuals in population
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.1 # probability for mutating an individual
MAX_GENERATIONS = 50 # max  number of generations for stopping condition


# As we are experminting with this code we would like the same experiment several times and get repeatable result, to do this we add a set seed value. Please not you would not normally do this in a normal run.
RANDOM_SEED = 42 
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint,0 ,1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("individualCreator",tools.initRepeat,creator.Individual,toolbox.zeroOrOne,ONE_MAX_LENGTH)
toolbox.register("populationCreator",tools.initRepeat,list,toolbox.individualCreator)

def oneMaxFitness(individual):
  return sum(individual), #return a tuple

toolbox.register("evaluate", oneMaxFitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate",tools.mutFlipBit,indpb=1.0/ONE_MAX_LENGTH)
population = toolbox.populationCreator(n=POPULATION_SIZE)
generationCounter = 0
fitnessValue = list(map(toolbox.evaluate,population))
for individual, fitnessValue in zip(population,fitnessValue):
  individual.fitness.values = fitnessValue
fitnessValues = [individual.fitness.values[0] for individual in population]
maxFitnessValues = []
meanFitnessValues = []

while max(fitnessValue) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
  generationCounter+=1
  offspring = toolbox.select(population, len(population))
  offspring = list(map(toolbox.clone, offspring))
  for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < P_CROSSOVER:
      toolbox.mate(child1,child2)
      del child1.fitness.values
      del child2.fitness.values

  for mutant in offspring:
    if random.random() < P_MUTATION:
      toolbox.mutate(mutant)
      del mutant.fitness.values
  newIndividuals = [ind for ind in offspring if not ind.fitness.valid]
  newFitnessValues = list(map(toolbox.evaluate,newIndividuals))
  for individual, fitnessValue in zip(newIndividuals,newFitnessValues):
    individual.fitness.values = fitnessValue
  population[:] = offspring


  fitnessValues = [ind.fitness.values[0] for ind in population]
  maxFitness = max(fitnessValues)
  meanFitness = sum(fitnessValues)/ len(population)
  maxFitnessValues.append(maxFitness)
  meanFitnessValues.append(meanFitness)
  print("- Generation {}: Max Fitness = {}, Avg Fitness = {}"
      .format(generationCounter, maxFitness, meanFitness))
  best_index = fitnessValues.index(max(fitnessValues))
  print("Best Individual = ", *population[best_index], "\n")

""" Plotting results """
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()