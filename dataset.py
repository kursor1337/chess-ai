import os

from chess import pgn

import numpy as np
from chess import Board
from chess import Move
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam # type: ignore
import time
import pandas as pd


def board_to_matrix(board: Board):
    matrix = np.zeros((8, 8, 12))
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[row, col, piece_type + piece_color] = 1
    return matrix

def encode_move(move: Move):
    return move.from_square * 64 + move.to_square

files = [file for file in os.listdir("data") if file.endswith(".pgn")]

def chess_train_data_generator():
    for file in files[:-1]:
        file_path = f"data/{file}"
        with open(file_path, 'r') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():

                    x = board_to_matrix(board)
                    y = encode_move(move)
                    yield x, y

                    board.push(move)

def chess_validation_data_generator():
    file_path = f"data/{files[-1]}"
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():

                x = board_to_matrix(board)
                y = encode_move(move)
                yield x, y

                board.push(move)
