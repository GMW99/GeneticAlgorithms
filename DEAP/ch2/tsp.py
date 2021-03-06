import csv
import pickle
import os
import codecs

import numpy as np

from urllib.request import urlopen

import matplotlib.pyplot as plt

class TravelingSalesmanProblem:

    def __init__(self,name):
        """
        Creates an instance of the TSP problem

        :param name: Name of TSP problem
        """
        
        # initializing instance variables
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        # initializing the data
        self.__initData()
        
    
    def __len__(self):
        """
        :return: the length of TSP (number of cities)
        """
        return self.tspSize
    
    def __initData(self):
        """Reads the serialized data, and if not available - calls __create_data() to prepare it
        """

        # attempt to read serialized data
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass

        # serailized data not found - create the data instead
        if not self.locations or not self.distances:
            self.__createData()

        # set the problem 'size'
        self.tspSize = len(self.locations)
    
    def __createData(self):
        """
        Reads the desored TSP file from the interent, gets the coordinates of the cities
        and then calculates the distances between all the cities. With that data it 
        populates a matrix. Then serializes the city location and distances, this is done
        using pickle.
        """
        self.locations = []

        with urlopen("http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/" + self.name + ".tsp") as file:
            reader = csv.reader(codecs.iterdecode(file,'utf-8'), delimiter=" ", skipinitialspace=True)

            # skip lines till it finds an line including
            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break
            # read till end of file (EOF) 
            for row in reader:
                if row[0] != 'EOF':
                    #remove index at beginning of line
                    del row[0]

                    # convert x,y coordinates to a nparray
                    self.locations.append(np.asarray(row, dtype=np.float))
                else:
                    break

            #set the problem size
            self.tspSize = len(self.locations)

            # print data
            print("length = {}, locations = {}".format(self.tspSize, self.locations))

            # initialize distance matrix by filling it with 0's
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

            # populate the distance matrix with calculated distances
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # calculate euclidean distance between two ndarrays 
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j], distance))

            # serialize locations and distances
            if not os.path.exists("tsp-data"):
                os.makedirs("tsp-data")
            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))

    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by the given indices.
        :param indices: A list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices.
        """
        # distance between the last and first city
        distance = self.distances[indices[-1]][indices[0]]

        # add the distance between each pair of consequtive cities
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]
        return distance
    
    def plotData(self,indices):
        """
        plots the path described by the given indices of the cities
        :param indices: A list of ordered city indices describing the given path.
        :param title: A title to be displayed on the graph
        :return: the resulting plot
        """
        plt.title(self.name)
        # plot the dots representing the cities
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # create a list of the corresponding city locations
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # plot a line between each pair of consequtive cities
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt