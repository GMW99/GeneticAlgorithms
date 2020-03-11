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
    def __len__(self):
        """
        :return: The number of features used in the classification.
        """
        return self.x.shape[1]
    
    def getMeanAccuracy(self,zeroOneList):
        """
        returns the mean accuracy of the classifier, which is calcualted for the test set,
        after the training using the features selected in the zeroOneList.
        :param zeroOneList: a list of of 0's or 1's that coresponed to features in the dataset. Where 0 indicated not used
        and vis versa for 1. 
        :return: the mean accurary of the classifier when using the features in zeroOneList.
        """

        # drop the columns of the training and test sets that 
        # correspond to unselected features.

        zeroIndices = [i for i, n in enumerate(zeroOneList) if n ==0]
 
        currentXTrain = self.x.drop(self.x.columns[zeroIndices], 1)
    
        # train the regerssor 
        cvResults = model_selection.cross_val_score(self.classifier, currentXTrain, self.y, cv=self.kfold, scoring='accuracy')

        return cvResults.mean()
        