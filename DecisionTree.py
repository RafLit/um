#!/usr/bin/env python3
##
# @author Rafal Litka
# @file DecisionTree.py

from Classifier import Classifier
import math
import pandas as pd
import random
import ete3

##  Wezel drzewa
class TreeNode(Classifier):

    def __init__(self):
        super().__init__()
        self.leafes = {}
        self.branches = {}


    def fit(self, trainSet, trainLab):
        entropies = {feature:0 for feature in trainSet.columns}
        for feature in trainSet.columns:
            column = trainSet[feature]
            for value in column.unique():
                reducedLab = trainLab[column==value]
                # wyznaczanie entropii
                entropy = 0.
                for count in reducedLab.value_counts():
                    x = count/len(reducedLab)
                    if not (x == 0. or x == 1.):
                        entropy -= x*math.log2(x)
                entropies[feature] += len(reducedLab)/len(trainLab)*entropy

        # wybor cechy o minimalnej entropii
        self.splitFeature = min(entropies,key=entropies.get)
        column = trainSet[self.splitFeature]
        for value in trainSet[self.splitFeature].unique():

            reducedSet = trainSet.loc[column == value, :]
            reducedLab = trainLab[column == value]
            # obliczanie entropii
            entropy = 0.
            for count in reducedLab.value_counts():
                x = count / len(reducedLab)
                if not (x == 0. or x == 1.):
                    entropy -= x * math.log2(x)

            if entropy == 0.0 or len(trainSet.columns) == 1:
                # tworzenie liscia
                self.leafes[value] = TreeLeaf()
                self.leafes[value].fit(reducedLab)
            else:
                # tworzenie drzewa
                self.branches[value] = TreeNode()
                self.branches[value].fit(reducedSet.drop(labels=[self.splitFeature],axis='columns'), reducedLab)


    def predict(self, testSet):
        values = testSet.loc[:,self.splitFeature].unique()
        tLab = pd.Series(None,index=testSet.index, name='class')
        for value in values:
            if value in list(self.leafes.keys()) + list(self.branches.keys()):
                if value in list(self.leafes.keys()):
                    tLab[testSet[self.splitFeature] == value] = self.leafes[value].predict()
                if value in list(self.branches.keys()):
                    reducedSet = testSet.loc[testSet.loc[:,self.splitFeature] == value,:]
                    tLab[testSet[self.splitFeature] == value] = self.branches[value].predict(reducedSet)
            else:
                # losowy wybor jezeli atrybutu nie bylo w zbiorze trenujacym
                randChoice = random.choice(list(self.leafes.keys()) + list(self.branches.keys()))
                if randChoice in list(self.leafes.keys()):
                    tLab[testSet[self.splitFeature] == value] = self.leafes[randChoice].predict()
                else:
                    reducedSet = testSet.loc[testSet.loc[:,self.splitFeature] == value,:]
                    tLab[testSet[self.splitFeature] == value] = self.branches[randChoice].predict(reducedSet)
        return tLab

    ## Przycinanie drzewa
    # @param pruneSet zestaw danych do przyciecia drzewa
    # @param pruneLab zestaw etykiet do przyciecia drzewa
    def prune(self, pruneSet, pruneLab):
        toDel = []
        for value in self.branches.keys():
            reducedSet = pruneSet.loc[pruneSet.loc[:, self.splitFeature] == value, :]
            if not reducedSet.shape[0]:
                continue
            reducedLab = pruneLab[pruneSet[self.splitFeature] == value]
            nodeRes = self.branches[value].predict(reducedSet)
            nodeErr = sum(nodeRes != reducedLab)
            leaf = TreeLeaf()
            leaf.fit(reducedLab)
            leafRes = pd.Series(None, index=reducedLab.index, name=reducedLab.name)
            leafRes[:] = leaf.predict()
            leafErr = sum(leafRes != reducedLab)
            if leafErr < nodeErr:
                self.leafes[value] = leaf
                toDel.append(value)
            else:
                self.branches[value].prune(reducedSet, reducedLab)
        for d in toDel:
            del self.branches[d]

    ## wizualizacja drzewa
    def visualize(self):
        me = ete3.TreeNode(name=self.splitFeature)
        for key in self.leafes.keys():
            me.add_child(self.leafes[key].visualize())
        for key in self.branches.keys():
            me.add_child(self.branches[key].visualize())
        return me






## lisc drzewa
class TreeLeaf:

    def __init__(self):
        super().__init__()
        self.decision = None

    def fit(self,  trainLab):
        counts = trainLab.value_counts()
        self.decision = counts.index[counts.argmax()]

    def predict(self):
        return self.decision

    ## wizualizacja liscia
    def visualize(self):
        return ete3.TreeNode(name=self.decision)

## Drzewo decyzyjne
class DecisionTree(Classifier):


    def __init__(self, pruneSplit = 0.0):
        super().__init__()
        self.rootNode = TreeNode()
        self.pruneSplit = pruneSplit

    def fit(self, trainSet, trainLab):
        n = trainLab.shape[0]
        split = int(self.pruneSplit*n)
        pruneSet = trainSet.iloc[:split]
        pruneLab = trainLab.iloc[:split]
        trainSet = trainSet.iloc[split:]
        trainLab = trainLab.iloc[split:]
        self.rootNode.fit(trainSet, trainLab)
        self.rootNode.prune(pruneSet, pruneLab)

    def predict(self, testSet):
        return self.rootNode.predict(testSet)

    ## Przycinanie drzewa
    def prune(self, pruneSet, pruneLab):
        self.rootNode.prune(pruneSet, pruneLab)

    ## Wizualizacja drzewa
    def visualize(self, fileName):
        t = self.rootNode.visualize()
        ts = ete3.TreeStyle()
        def my_layout(node):
            if node.is_leaf():
                # If terminal node, draws its name
                name_face = ete3.AttrFace("name")
            else:
                # If internal node, draws label with smaller font size
                name_face = ete3.AttrFace("name", fsize=10)
            # Adds the name face to the image at the preferred position
            ete3.faces.add_face_to_node(name_face, node, column=0, position="branch-right")
        ts.layout_fn = my_layout
        ts.show_leaf_name = False
        t.render(file_name=fileName, tree_style=ts)
        pass

