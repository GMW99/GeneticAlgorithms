from deap import tools
from deap import algorithms
import numpy as np

def selTournamentWithSharing(individuals, k, tournsize, fit_attr="fitness",distanceThreshold = 0.1, sharingExtent = 5.0):

    # get original fitness values
    originalFitness = [ind.fitness.values[0] for ind in individuals]

    # apply sharing to each individual:
    for i in range(len(individuals)):
        sharingSum = 1

        # iterate over all other individuals
        for j in range(len(individuals)):
            if i != j:
                # calculate eucledean distance between individuals:
                distance = np.sqrt(
                    ((individuals[i][0] - individuals[j][0]) ** 2) + ((individuals[i][1] - individuals[j][1]) ** 2))

                if distance < distanceThreshold:
                    sharingSum += (1 - distance / (distanceThreshold * sharingExtent))

        # reduce fitness accordingly:
        individuals[i].fitness.values = originalFitness[i] / sharingSum,

    # apply original tools.selTournament() using modified fitness:
    selected = tools.selTournament(individuals, k, tournsize, fit_attr)

    # retrieve original fitness:
    for i, ind in enumerate(individuals):
        ind.fitness.values = originalFitness[i],

    return selected