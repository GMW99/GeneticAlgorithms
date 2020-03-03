import numpy as np


class Knapsack01Problem:
    """
    Class to encapsulate the Knapsack01Problem as defined from 
    RosettaCode.org
    """
    def __init__(self):
        # instance variables
        self.items = []
        self.maxCapacity = 0

        # data
        self.__initData()

    def __len__(self):
        """
        :return: the amount of items defined by the problem.
        """
        return len(self.items)

    def __initData(self):
        """
        Initialisation of the RosettaCode knapsack 0-1 problem data.
        """
        self.items = [
            ("map", 9, 150),
            ("compass", 13, 35),
            ("water", 153, 200),
            ("sandwich", 50, 160),
            ("glucose", 15, 60),
            ("tin", 68, 45),
            ("banana", 27, 60),
            ("apple", 39, 40),
            ("cheese", 23, 30),
            ("beer", 52, 10),
            ("suntan cream", 11, 70),
            ("camera", 32, 30),
            ("t-shirt", 24, 15),
            ("trousers", 48, 10),
            ("umbrella", 73, 40),
            ("waterproof trousers", 42, 70),
            ("waterproof overclothes", 43, 75),
            ("note-case", 22, 80),
            ("sunglasses", 7, 20),
            ("towel", 18, 12),
            ("socks", 4, 50),
            ("book", 30, 10)
        ]

        self.maxCapacity = 400

    def getValue(self,zeroOneList):
        """
        Calculates the value of the selected items in the list, while
        ignoring items that will cause the sack to go over the allowed weight.
        :param zeroOneList: a list of 1/0 values corrosponding to the items
        on the list with 1 meaning in knapsack.
        :return: the total value that has been calculated.
        """
        totalWeight = totalValue = 0
        
        for i in range(len(zeroOneList)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                totalWeight += zeroOneList[i] * weight
                totalValue += zeroOneList[i] * value
        return totalValue

    def printItems(self,zeroOneList):
        """
        Prints the item selected in the list, while
        ignoring items that will cause the sack to go over the allowed weight.
        :param zeroOneList: a list of 1/0 values corrosponding to the items
        on the list with 1 meaning in knapsack.
        """
        totalWeight = totalValue = 0
        
        for i in range(len(zeroOneList)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                if zeroOneList[i] > 0:
                    totalWeight += weight
                    totalValue += value
                    print("- Adding {}: weight = {}, value = {}, accumulated weight = {},accumulated value = {}"
                        .format(item, weight, value, totalWeight, totalValue))
        print("- Total weight = {}, Total value = {}".format(totalWeight, totalValue))