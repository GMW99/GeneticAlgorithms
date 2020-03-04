import csv
import pickle
import os
import codecs

import numpy as np

from urllib.request import urlopen

import matplotlib.pyplot as plt

class TravellingSalesmanProblem:

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
        pass
    
    def __initData(self):
        pass
    
    def __createData(self):
        pass

    def getTotalDistance(self):
        pass
    
    def plotData(self):
        pass
