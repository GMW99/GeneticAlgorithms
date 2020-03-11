#Maths library
import numpy as np

# Data science library
from sklearn import model_selection
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Ploting library
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

class Friedman1:

    TEST_SIZE = 0.20
    NOISE = 1.0

    def __init__(self,numFeatures=5, numSamples=60, randomSeed=42):
        """
        :param numFeatures: Amount of features in the dataset will be set to 5 if lower value entered.
        :param numSamples: Amount of samples in the dataset.
        :param randomSeed: Forces repeatable results.
        """


        self.numFeatures = setNumFeatures(numFeatures)
        self.numSamples = numSamples
        self.randomSeed = randomSeed
        # create data
        self.x, self.y = datasets.make_friedman1(n_samples=self.numSamples,n_features=self.numFeatures,noise=self.NOISE, random_state=self.randomSeed)

        # split data into training and test sets.
        self.xTrain, self.xTest, self.yTrain, self.yTest = \
            model_selection.train_test_split(self.x, self.y, test_size=self.TEST_SIZE, random_state=self.randomSeed)

        self.regressor = GradientBoostingRegressor(random_state=self.randomSeed)

    def setNumFeatures(self, numFeatures):
        if numFeatures < 5:
            return 5
        else:
            return numFeatures

    def __len__(self):
        """
        :return: the amount of features
        """
        return self.numFeatures