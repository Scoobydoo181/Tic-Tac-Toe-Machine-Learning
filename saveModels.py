'''
Saves three regression models into file objects for later loading
'''

import numpy as np
from sklearn import svm, neighbors, neural_network
from joblib import dump

#Local import from validate.py
from validate import LinearRegression

#Dataset representing next-best move in the middle of a game
bestMoveMultiData = np.loadtxt('./datasets/tictac_multi.txt', dtype=int)

xs = bestMoveMultiData[:, :9]  # 6551x9
ys = bestMoveMultiData[:, 9:]  # 6551x9

regressor1 = neighbors.KNeighborsRegressor(weights='distance')
regressor1.fit(xs, ys)

dump(regressor1, './savedModels/knn.model')

regressor2 = LinearRegression()
regressor2.fit(xs, ys)
dump(regressor2, './savedModels/linreg.model')

regressor3 = neural_network.MLPRegressor(max_iter=1000)
regressor3.fit(xs, ys)
dump(regressor3, './savedModels/mlp.model')



