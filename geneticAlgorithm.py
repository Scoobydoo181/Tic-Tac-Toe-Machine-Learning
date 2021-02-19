import pygad
import pygad.kerasga
import numpy as np
from tensorflow import keras

from joblib import dump

from play import gameOver

from numpy.random import default_rng
rng = default_rng()

model = keras.Sequential()
model.add(keras.layers.Input(9))
model.add(keras.layers.Dense(9, activation='relu'))
model.add(keras.layers.Dense(1, activation='relu'))

keras_ga = pygad.kerasga.KerasGA(model, num_solutions=10)

def boardScore(board, move): 
    #Through matrix multiplicaiton with the board vector, generate the sum over every possible 3 square line
    score = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      [1, 0, 0, 1, 0, 0, 1, 0, 0],

                      [1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 1, 0],

                      [1, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 0, 1, 0, 1, 0, 1, 0, 0],

                      [1, 0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0],

                      [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 1, 0, 1, 0, 0],

                      [0, 0, 1, 1, 1, 2, 0, 0, 1],
                      [0, 0, 1, 1, 1, 2, 0, 0, 1],

                      [0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [1, 0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0, 1, 0, 0],

                      [0, 0, 0, 0, 1, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1, 0, 0, 1, 0],

                      [0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 1, 0, 0, 1]])
    
    #Transform the board such that the move to make is 1, the opponent is -1, and an empty square is 0.5
    modifiedBoard = -1*np.array([-0.5 if val == 0 else val for val in board])
    modifiedBoard[move] = 1

    #Through matrix multiplication, sum up the scores for each 3 square line per square
    #i.e. 3 ways to win for a corner, 2 ways to win for an edge, 4 ways to win for the center
    combine = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    
    #Transform each sum of scores into a weighted value such that each move is 4x more important than the next most important move
    scoreMap = {
        3: 7 ** 7,
        -1: 6 ** 6,
        2.5: 5 ** 5,
        2: 4 ** 4,
        1: 3 ** 3,
        0.5: 2 ** 2
    }
    #Perform the transformations and get the raw score (sum of grid values for each way to win) for the chosen move
    transformedScores = combine @ np.array([scoreMap.setdefault(val, 0) for val in (score @ modifiedBoard)])
    
    return transformedScores[move]

def randomGameBoard():
    #Generates a random game board by radomly choosing an available move for X, 
    # then choosing the move with the highest fitness value for O
    board = np.zeros(9)
    for i in range(rng.integers(low=0, high=9)):
        if i % 2 == 0:
            move = rng.choice([i for i in range(9) if board[i] == 0])
            board[move] = 1
        else:
            score, move = max( (boardScore(board, move), move) for move in range(9) if board[move] == 0)
            board[move] = -1
    return board

def fitnessFunc(solution, index):
    #Implementation of the tic-tac-toe heuristic from the academic paper
    weightsMatrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)

    model.set_weights(weights=weightsMatrix)

    board = randomGameBoard()

    move = round(model.predict(board.reshape((1, 9)))[0,0])

    score = boardScore(board, move) if move in range(9) and board[move] == 0 else 0

    allScores = [boardScore(board, nextMove) for nextMove in range(9) if board[nextMove] == 0]
    maxScore = max(allScores)

    if maxScore == 0:
        return 0
    return score / maxScore



if __name__ == '__main__':
    ga_instance = pygad.GA(num_generations=50000,
                        num_parents_mating=2,
                        initial_population=keras_ga.population_weights,
                        fitness_func=fitnessFunc,
                        parent_selection_type='rank', #roulette wheel selection
                        crossover_type='uniform', #Randomly select attributes from both parents
                        crossover_probability=0.8,
                        mutation_type='random',
                        mutation_probability=0.1)

    ga_instance.run()
    print('Finished running')


    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    bestWeights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(bestWeights)
    bestMoveSingleData = np.loadtxt('./datasets/tictac_single.txt', dtype=int)

    xs = bestMoveSingleData[:, :9]
    ys = bestMoveSingleData[:, 9]

    print(xs.shape, ys.shape)
    model.compile(loss="mean_squared_error", metrics=['accuracy'])
    accuracy = model.evaluate(x=xs, y=ys, batch_size=6551)
    print('Accuracy: ', accuracy)

    dump(model, './genetic.model')

    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
