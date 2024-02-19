from sklearn import tree
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import random
import math
import numpy as np
import time
import heapq
import enchant
from sklearn.metrics import precision_score, recall_score, f1_score


class Statistics:

    def __init__(self, prediction, target, num_classes):
        self.prediction = prediction
        self.confusionMatrix = np.zeros((num_classes, num_classes))
        for i in range(len(prediction)):
            self.confusionMatrix[target[i], prediction[i]] += 1
        self.accuracy = (np.trace(self.confusionMatrix)/np.sum(self.confusionMatrix))*100
        self.accuracyClass = np.zeros(num_classes)
        for i in range(num_classes):
            self.accuracyClass[i] = (self.confusionMatrix[i,i]/np.sum(self.confusionMatrix[i,:]))*100
        self.f1 = f1_score(target, prediction,average='weighted')
        self.precision = precision_score(target, prediction,average='weighted')
        self.recall = recall_score(target, prediction,average='weighted')


class RandomForest:

    def __init__(self, numberOfTrees, criterion_input, parameter, numClasses):
        self.size = numberOfTrees
        self.num_classes = numClasses
        self.ensemble_method = parameter
        self.criterion = criterion_input
        self.trees = []
        self.statisticsTreesValidation = []
        self.statisticsRFValidation = []
        self.statisticsTrees = []
        self.statisticsRF = []
        self.timeTraining = 0
        self.timeValidation = 0
        self.timeTesting = 0

        for i in range(self.size):
            self.trees.append(tree.DecisionTreeClassifier(criterion = criterion_input))

    def fit(self, Data, target, percentage):

        start = time.time()

        print("training 0%", end = "\r")
        for i in range(self.size):
            indexes = range(len(Data))
            indexes = random.sample(indexes, round(percentage * len(Data)))
            self.trees[i].fit(Data[indexes,:], target[indexes])
            print("training "+str(math.ceil(i/self.size))+"%", end = "\r")

        print("Training finished")

        end = time.time()
        self.timeTraining = end - start

    def validate(self, Data, target):

        start = time.time()

        self.statisticsTreesValidation = []
        for i in range(self.size):
            prediction = self.trees[i].predict(Data)
            self.statisticsTreesValidation.append(Statistics(prediction, target, self.num_classes))

        if self.ensemble_method == "SV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTreesValidation[i].prediction)):
                    results[j,self.statisticsTreesValidation[i].prediction[j]] += 1

            prediction = np.argmax(results, axis = 1)
            self.statisticsRFValidation = Statistics(prediction, target, self.num_classes)


        if self.ensemble_method == "WV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTreesValidation[i].prediction)):

                    results[j,self.statisticsTreesValidation[i].prediction[j]] += self.statisticsTreesValidation[i].accuracyClass[self.statisticsTreesValidation[i].prediction[j]] * np.average(self.statisticsTreesValidation[i].accuracyClass)

            prediction = np.argmax(results, axis = 1)
            self.statisticsRFValidation = Statistics(prediction, target, self.num_classes)


        #TODO: if self.ensemble_method == "DSTheory":

        end = time.time()
        self.timeValidation = end - start

    def predict(self, Data, target):

        start = time.time()

        self.statisticsTrees = []

        for i in range(self.size):
            prediction = self.trees[i].predict(Data)
            self.statisticsTrees.append(Statistics(prediction, target, self.num_classes))

        if self.ensemble_method == "SV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTrees[i].prediction)):
                    results[j,self.statisticsTrees[i].prediction[j]] += 1

            prediction = np.argmax(results, axis = 1)
            self.statisticsRF = Statistics(prediction, target, self.num_classes)


        if self.ensemble_method == "WV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTrees[i].prediction)):

                    results[j,self.statisticsTrees[i].prediction[j]] += self.statisticsTreesValidation[i].accuracyClass[self.statisticsTrees[i].prediction[j]] * np.average(self.statisticsTreesValidation[i].accuracyClass)

            prediction = np.argmax(results, axis = 1)
            self.statisticsRF = Statistics(prediction, target, self.num_classes)



        #TODO: if self.ensemble_method == "DSTheory":

        end = time.time()
        self.timeTesting = end - start

    def plot(self, columns, name):

        rows = math.ceil(self.size/columns)
        fig, axs = plt.subplots(rows, columns, dpi=3000)
        #fig.set_size_inches(20,16)
        printedTrees = 0
        for i in range(rows):
            for j in range(columns):
                if printedTrees < self.size:
                    plot_tree(RF.trees[printedTrees], filled=True, ax = axs[i, j])
                    printedTrees = printedTrees + 1
                    axs[i, j].set_title('Tree: ' + str(printedTrees), fontsize = 7)

        fig.savefig(name)
        #plt.show()



# def tree2StringV2(node):
#     children_left = tree.tree_.children_left
#     children_right = tree.tree_.children_right
#     feature = tree.tree_.feature
#     threshold = tree.tree_.threshold
#
#     stack = [(0,0,0)]
#     result = ""
#
#     while len(stack) > 0:
#
#         node_id, depth, side = stack.pop()
#         if depth != 0:
#             result = result + "("
#         result = result + str(node_id)
#         if children_left[node_id] != children_right[node_id]:
#             result = result + "("
#             stack.append((children_right[node_id], depth + 1,side+1))
#             stack.append((children_left[node_id], depth + 1,side))
#         else:
#             result = result + ")"
#             if side > 0:
#                 for i in range(0,side):
#                     for j in range(i):
#                         result = result + ")"
#     return result

def tree2String(tree, node_id, result):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    result = result + str(node_id)

    if children_left[node_id] != children_right[node_id]:
        result = result + "("
        result = tree2String(tree, children_left[node_id], result)
        result = result + ")"
        if children_right[node_id] > 0:
            result = result + "("
            result = tree2String(tree, children_right[node_id], result)
            result = result + ")"
    else:
        return result
    return result


class FederatedLearning:

    def __init__(self):
        self.size = 0
        self.num_classes = 0
        self.ensemble_method = 0
        self.criterion = 0
        self.trees = []
        self.statisticsTreesValidation = []
        self.statisticsRFTreeBelong = []
        self.statisticsRFValidation = []
        self.statisticsTrees = []
        self.statisticsRF = []
        self.timeM1 = 0
        self.timeM2 = 0
        self.timeM3 = 0
        self.timeM4 = 0
        self.timeTesting = 0

    # Method that add all trees from all RFs.
    def mergeMethod1(self, RFs, num_input):

        start = time.time()

        for i in range(len(RFs)):
            if i == 0:
                self.num_classes = RFs[i].num_classes
                self.ensemble_method = RFs[i].ensemble_method
                self.criterion = RFs[i].criterion
            self.size += RFs[i].size
            for j in range(RFs[i].size):
                self.trees.append(RFs[i].trees[j])
                self.statisticsTreesValidation.append(RFs[i].statisticsTreesValidation[j])
                self.statisticsRFTreeBelong.append(i)

        measure = []
        for j in range(self.size):
            measure.append(self.statisticsTreesValidation[j].accuracy)

        #idx = map(measure.index, heapq.nlargest(num_input, measure))
        best = heapq.nlargest(num_input, enumerate(measure), key=lambda x: x[1])

        auxsize = 0
        auxtrees = []
        auxstatisticsTreesValidation = []
        auxstatisticsRFTreeBelong = []
        for idx in best:
            auxsize += 1
            auxtrees.append(self.trees[idx[0]])
            auxstatisticsTreesValidation.append(self.statisticsTreesValidation[idx[0]])
            auxstatisticsRFTreeBelong.append(self.statisticsRFTreeBelong[idx[0]])

        self.size = auxsize
        self.trees = auxtrees
        self.statisticsTreesValidation = auxstatisticsTreesValidation
        self.statisticsRFTreeBelong = auxstatisticsRFTreeBelong

        end = time.time()
        self.timeM1 = end - start


    def mergeMethod2(self, RFs, num_input):

        start = time.time()

        for i in range(len(RFs)):
            if i == 0:
                self.num_classes = RFs[i].num_classes
                self.ensemble_method = RFs[i].ensemble_method
                self.criterion = RFs[i].criterion
            self.size += RFs[i].size
            for j in range(RFs[i].size):
                self.trees.append(RFs[i].trees[j])
                self.statisticsTreesValidation.append(RFs[i].statisticsTreesValidation[j])
                self.statisticsRFTreeBelong.append(i)

        measure = []
        for j in range(self.size):
            measure.append(self.statisticsTreesValidation[j].accuracy * np.average(self.statisticsTreesValidation[j].accuracyClass))

        #idx = map(measure.index, heapq.nlargest(num_input, measure))
        best = heapq.nlargest(num_input, enumerate(measure), key=lambda x: x[1])

        auxsize = 0
        auxtrees = []
        auxstatisticsTreesValidation = []
        auxstatisticsRFTreeBelong = []
        for idx in best:
            auxsize += 1
            auxtrees.append(self.trees[idx[0]])
            auxstatisticsTreesValidation.append(self.statisticsTreesValidation[idx[0]])
            auxstatisticsRFTreeBelong.append(self.statisticsRFTreeBelong[idx[0]])

        self.size = auxsize
        self.trees = auxtrees
        self.statisticsTreesValidation = auxstatisticsTreesValidation
        self.statisticsRFTreeBelong = auxstatisticsRFTreeBelong

        end = time.time()
        self.timeM2 = end - start

    def mergeMethod3(self, RFs, num_input):
        start = time.time()
        keepTreesIdx = []
        for i in range(len(RFs)):
            if i == 0:
                self.num_classes = RFs[i].num_classes
                self.ensemble_method = RFs[i].ensemble_method
                self.criterion = RFs[i].criterion
            measure = []
            for j in range(RFs[i].size):
                measure.append(RFs[i].statisticsTreesValidation[j].accuracy)

            #idx = map(measure.index, heapq.nlargest(num_input, measure))
            best = heapq.nlargest(num_input, enumerate(measure), key=lambda x: x[1])

            for idx in best:
                self.size += 1
                self.trees.append(RFs[i].trees[idx[0]])
                self.statisticsTreesValidation.append(RFs[i].statisticsTreesValidation[idx[0]])
                self.statisticsRFTreeBelong.append(i)

        end = time.time()
        self.timeM3 = end - start


    def mergeMethod4(self, RFs, num_input):
        start = time.time()
        keepTreesIdx = []
        for i in range(len(RFs)):
            if i == 0:
                self.num_classes = RFs[i].num_classes
                self.ensemble_method = RFs[i].ensemble_method
                self.criterion = RFs[i].criterion
            measure = []
            for j in range(RFs[i].size):
                measure.append(RFs[i].statisticsTreesValidation[j].accuracy * np.average(RFs[i].statisticsTreesValidation[j].accuracyClass))

            #idx = map(measure.index, heapq.nlargest(num_input, measure))
            best = heapq.nlargest(num_input, enumerate(measure), key=lambda x: x[1])

            for idx in best:
                self.size += 1
                self.trees.append(RFs[i].trees[idx[0]])
                self.statisticsTreesValidation.append(RFs[i].statisticsTreesValidation[idx[0]])
                self.statisticsRFTreeBelong.append(i)

        end = time.time()
        self.timeM4 = end - start

    def mergeMethod5(self, RFs):
        start = time.time()
        tree1 = tree2String(RFs[0].trees[0],0,"")
        tree2 = tree2String(RFs[0].trees[30],0,"")
        print(len(tree1))
        print(len(tree2))
        print(print(enchant.utils.levenshtein(tree1, tree2)))

        end = time.time()
        self.timeM5 = end - start
        print(self.timeM5)


    def predict(self, Data, target):

        start = time.time()
        self.statisticsTrees = []
        for i in range(self.size):
            prediction = self.trees[i].predict(Data)
            self.statisticsTrees.append(Statistics(prediction, target, self.num_classes))

        if self.ensemble_method == "SV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTrees[i].prediction)):
                    results[j,self.statisticsTrees[i].prediction[j]] += 1

            prediction = np.argmax(results, axis = 1)
            self.statisticsRF = Statistics(prediction, target, self.num_classes)


        if self.ensemble_method == "WV":
            results = np.zeros((len(target),self.num_classes))
            for i in range(self.size):
                for j in range(len(self.statisticsTrees[i].prediction)):

                    results[j,self.statisticsTrees[i].prediction[j]] += self.statisticsTreesValidation[i].accuracyClass[self.statisticsTrees[i].prediction[j]] * np.average(self.statisticsTreesValidation[i].accuracyClass)

            prediction = np.argmax(results, axis = 1)
            self.statisticsRF = Statistics(prediction, target, self.num_classes)
