import random

from pandas import read_csv

from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier 

class Zoo: 
    """ 
    This class encapsulates the Zoo classificiation problem.
    """
    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5

    def __init__(self, randomSeed):
        """
        :param randomSeed: this variable is used so we can have repeatable results.
        """
        self.randomSeed = randomSeed
        # Skip the header and animal name
        self.data = read_csv(self.DATASET_URL,header=None,usecols=range(1,18))

        # separate the inputs from outputs
        self.x = self.data.iloc[:,0:16]
        self.y = self.data.iloc[:,16]

        # split dataset, creating a group of training and testing set to be used in k-fold testing process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = DecisionTreeClassifier(random_state=self.randomSeed)
    