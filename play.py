import numpy as np
from math import floor

from joblib import load

def printBoard(board):
    board = ['X' if val == 1 else 'O' if val == -1 else '_' for val in board]
    print(" ".join(board[0:3]))
    print(" ".join(board[3:6]))
    print(" ".join(board[6:9]))

def gameOver(board, pos):
    marker = board[pos]

    rowOffset = pos % 3
    #Check column
    if board[rowOffset] == board[3 + rowOffset] and board[3 + rowOffset] == board[6 + rowOffset]:
        return True

    colStart = pos - rowOffset
    #Check row
    if board[colStart] == board[colStart + 1] and board[colStart + 1] == board[colStart + 2]:
        return True

    #Check diagonal
    diag = [0, 4, 8]
    if pos in diag and reduce(lambda a, b: a and b, (board[i] == board[pos] for i in diag)):
        return True

    diag = [2, 4, 6]
    if pos in diag and reduce(lambda a, b: a and b, (board[i] == board[pos] for i in diag)):
        return True
    
    return False

#Human is X (+1), computer is O (-1)
def ticTacToe():
    board = np.zeros(9, dtype=int)

    #Load the models saved by saveModels.py
    #Choices are:
    #1. k-nearest-neighbors = './savedModels/knn.model'
    #2. Linear Regression = './savedModels/linreg.model'
    #3. Multilayer Perceptron regression = './savedModels/mlp.model'
    model = load('./savedModels/knn.model')

    print('Welcome to Tic-Tac-Toe!')
    print('You are player X, enter the number of the square you want to choose where 1 is the top left and 9 is the bottom right')
    draw = True
    roundCount = 0
    while roundCount < 5:
        printBoard(board)

        pos = int(input('(1-9)> ')) - 1

        if board[pos] != 0:
            continue
        
        board[pos] = 1

        if gameOver(board, pos):
            draw = False
            break
        
        print('Computer\'s turn:')

        if roundCount ==  4:
            break
        
        a = model.predict(board.reshape(1,-1))
        b = np.abs(board.reshape(1, -1))
        
        pos = np.argmax(a - b)
        board[pos] = -1

        if gameOver(board, pos):
            draw = False
            break
        
        roundCount = roundCount + 1

    printBoard(board)
    print('Draw' if draw else 'Congratulations! You won!' if board[pos] == 1 else 'Sorry. You lost.')

if __name__ == '__main__':
    ticTacToe()
