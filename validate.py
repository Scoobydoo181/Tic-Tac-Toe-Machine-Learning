""" 
There are three datasets:
    1. Final state of a game (predict winner)
    2. Intermediate state of a game (predict next best move-single)
    3. Intermediate state of a game (predict next best moves-multiple)

A linear SVM, k-nearest neighbors, and multilayer perceptron are each trained and evaluated on 1 and 2
A confusion matrix is output to evaluate each classifier on 1 and 2
Confusion matrixes display error in the form: (true positive, false positive, true negative, false negative)

k-nearest neighbors, linear regression, and multilayer perceptron algorithms are each trained an evaluated on 3

Classifiers are trained on 1 and 2
Regressors are trained on 3
"""

import numpy as np
from sklearn import svm, neighbors, neural_network
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump, load

# Suppress scientific notation
np.set_printoptions(suppress=True)

rng = np.random.default_rng()

def computeMetrics(model, xs, ys, skipConfusion=False):
    kf = KFold(n_splits=10, shuffle=True)

    predicted = None
    actual = None

    for train_index, test_index in kf.split(xs, ys):
        x_train, x_test = xs[train_index], xs[test_index]
        y_train, y_test = ys[train_index], ys[test_index]

        model.fit(x_train, y_train)

        y_predicted = model.predict(x_test)

        if predicted is None and actual is None:
            predicted = y_predicted
            actual = y_test 
        else:
            predicted = np.concatenate((predicted, y_predicted), axis= 0)
            actual = np.concatenate((actual, y_test), axis= 0)

    confusionMatrix = None if skipConfusion else confusion_matrix(actual, predicted, normalize='true')
    accuracy = accuracy_score(
        actual.reshape(-1, 1) if skipConfusion else actual, 
        np.round(predicted).reshape(-1, 1) if skipConfusion else predicted
        )

    return accuracy, confusionMatrix

def validateVictory():
    #Dataset representing endgame states (after a player has won)
    victoryData = np.loadtxt('./datasets/tictac_final.txt', dtype=int)

    xs = victoryData[:, :8]
    ys = victoryData[:, 9]

    #Reduce dataset size to 1/10
    # rng.shuffle(victoryData)
    # [xs, *_] = np.array_split(victoryData[:, :8] , 10) # 958x8
    # [ys, *_] = np.array_split(victoryData[:, 9] , 10) # 958x1

    print("Victory dataset:")
    #Linear SVM
    victoryClassifier1 = svm.LinearSVC()
    accuracy, confusionMatrix = computeMetrics(victoryClassifier1, xs, ys)
    print('The accuracy of the Linear SVM Classifier is: ', accuracy)
    print('The confusion matrix for the Linear SVM Classifier is: \n', confusionMatrix)

    #k-nearest-neighbors
    victoryClassifier2 = neighbors.KNeighborsClassifier()
    accuracy, confusionMatrix = computeMetrics(victoryClassifier2, xs, ys)
    print('The accuracy of the K-Nearest-Neighbors Classifier is: ', accuracy)
    print('The confusion matrix for K-Nearest-Neighbors Classifier is: \n', confusionMatrix)

    #multilayer perceptron
    victoryClassifier3 = neural_network.MLPClassifier(max_iter=1000, activation='tanh')
    accuracy, confusionMatrix = computeMetrics(victoryClassifier3, xs, ys)
    print('The accuracy of the Multilayer Perceptron Classifier is: ', accuracy)
    print('The confusion matrix for the Multilayer Perceptron Classifier is: \n', confusionMatrix)


def validateBestMoveSingle():
    #Dataset representing endgame states (after a player has won)
    bestMoveSingleData = np.loadtxt('./datasets/tictac_single.txt', dtype=int)

    xs = bestMoveSingleData[:, :8]
    ys = bestMoveSingleData[:, 9]

    #Reduce dataset size to 1/10
    # rng.shuffle(bestMoveSingleData)
    # [xs, *_] = np.array_split(bestMoveSingleData[:, :8] , 10) # 6551x8
    # [ys, *_] = np.array_split(bestMoveSingleData[:, 9] , 10) # 6551x1

    #Add random noise to 3000 of the datapoints
    # ys[2551:5551] = ys[2551:5551] * rng.random() + (8 - ys[2551:5551]) * rng.random()

    print("Best move (single) dataset:")
    #Linear SVM
    bestMoveSingleClassifier1 = svm.LinearSVC()
    accuracy, confusionMatrix = computeMetrics(bestMoveSingleClassifier1, xs, ys)
    print('The accuracy of the Linear SVM Classifier is: ', accuracy)
    print('The confusion matrix for the Linear SVM Classifier is: \n', confusionMatrix)

    #k-nearest-neighbors
    bestMoveSingleClassifier2 = neighbors.KNeighborsClassifier(weights='distance')
    accuracy, confusionMatrix = computeMetrics(bestMoveSingleClassifier2, xs, ys)
    print('The accuracy of the K-Nearest-Neighbors Classifier is: ', accuracy)
    print('The confusion matrix for the K-Nearest-Neighbors Classifier is: \n', confusionMatrix)

    #multilayer perceptron
    bestMoveSingleClassifier3 = neural_network.MLPClassifier(max_iter=1000)
    accuracy, confusionMatrix = computeMetrics(bestMoveSingleClassifier3, xs, ys)
    print('The accuracy of the Multilayer Perceptron Classifier is: ', accuracy)
    print('The confusion matrix for the Multilayer Perceptron Classifier is: \n', confusionMatrix)

#Class for manual linear regression using normal equations
class LinearRegression:
    def fit(self, xs, ys):
        #Matrix equation: ys = xs*weights
        #Regression equation: xs'*ys = xs'*xs*weights
        self.weights = []

        #Add bias term to data
        N = xs.shape[0]
        xs = np.hstack(( xs, np.ones(N).reshape(-1, 1) ))

        for i in range(9):        
            self.weights.append(np.linalg.solve(np.transpose(xs) @ xs, np.transpose(xs) @ ys[:, i]))

    def predict(self, xs):
        #Add bias term to data
        N = xs.shape[0]
        xs = np.hstack(( xs, np.ones(N).reshape(-1, 1) ))

        return np.transpose(np.array([xs @ self.weights[i] for i in range(9)]))

def validateBestMoveMulti():
    #Dataset representing next-best move in the middle of a game
    bestMoveMultiData = np.loadtxt('./datasets/tictac_multi.txt', dtype=int)

    xs = bestMoveMultiData[:, :9]
    ys = bestMoveMultiData[:, 9:]

    #Reduce dataset size to 1/10
    # rng.shuffle(bestMoveMultiData)
    # [xs, *_] = np.array_split(bestMoveMultiData[:, :9], 10) #6551x9
    # [ys, *_] = np.array_split(bestMoveMultiData[:, 9:], 10) #6551x9

    print("Best move (multi) dataset:")
    #k-nearest neighbors
    bestMoveMultiRegressor1 = neighbors.KNeighborsRegressor(weights='distance')
    accuracy, _ = computeMetrics(bestMoveMultiRegressor1, xs, ys, skipConfusion=True)
    print('The accuracy of the K-Nearest-Neighbors Regressor is: ', accuracy)
    
    #linear regression (with normal equation)
    bestMoveMultiRegressor2 = LinearRegression()
    accuracy, _ = computeMetrics(bestMoveMultiRegressor2, xs, ys, skipConfusion=True)
    print('The accuracy of the Linear Regression Model is: ', accuracy)

    #multilayer perceptron
    bestMoveMultiRegressor3 = neural_network.MLPRegressor(max_iter=1000)
    accuracy, _ = computeMetrics(bestMoveMultiRegressor3, xs, ys, skipConfusion=True)
    print('The accuracy of the Multilayer Perceptron Regressor is: ', accuracy)

if __name__ == '__main__':
    validateVictory()
    validateBestMoveSingle()
    validateBestMoveMulti()
