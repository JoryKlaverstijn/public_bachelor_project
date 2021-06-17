import random
import numpy as np

letters = np.array([# A
            [[1, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0]]
            ,  # B
            [[0, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # C
            [[1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [1, 0, 0, 0, 0]]
            ,  # D
            [[0, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # E
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0]]
            ,  # F
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1]]
            ,  # G
            [[1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 1, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [1, 0, 0, 0, 1]]
            ,  # H
            [[0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0]]
            ,  # I
            [[1, 0, 0, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 0, 0, 0, 1]]
            ,  # J
            [[1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [1, 0, 0, 0, 1]]
            ,  # K
            [[0, 1, 1, 0, 1],
             [0, 1, 0, 1, 1],
             [0, 0, 1, 1, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 0, 1]]
            ,  # L
            [[0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0]]
            ,  # M
            [[1, 0, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0]]
            ,  # N
            [[0, 1, 1, 1, 0],
             [0, 0, 1, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0],
             [0, 1, 1, 1, 0]]
            ,  # O
            [[1, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [1, 0, 0, 0, 1]]
            ,  # P
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1]]
            ,  # Q
            [[1, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # R
            [[0, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1],
             [0, 1, 1, 0, 1],
             [0, 1, 1, 1, 0]]
            ,  # S
            [[1, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # T
            [[0, 0, 0, 0, 0],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1]]
            ,  # U
            [[0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [1, 0, 0, 0, 1]]
            ,  # V
            [[0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 1, 0, 1, 1]]
            ,  # W
            [[0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 1, 0, 1]]
            ,  # X
            [[0, 1, 1, 1, 0],
             [1, 0, 1, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 0, 1, 0, 1],
             [0, 1, 1, 1, 0]]
            ,  # Y
            [[0, 1, 1, 1, 0],
             [1, 0, 1, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1]]
            ,  # Z
            [[0, 0, 0, 0, 0],
             [1, 1, 1, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 0, 1, 1, 1],
             [0, 0, 0, 0, 0]]
            ])

digits = np.array([# 1
            [[1, 1, 0, 1, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 1, 0, 1, 1]]
            ,  # 0
            [[1, 1, 0, 1, 1],
             [1, 0, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 0, 1, 1],
             [1, 0, 0, 0, 1]]
            ,  # 1
            [[1, 0, 0, 0, 1],
             [0, 1, 1, 1, 0],
             [1, 1, 0, 0, 1],
             [1, 0, 1, 1, 1],
             [0, 0, 0, 0, 0]]
            ,  # 3
            [[0, 0, 0, 0, 1],
             [1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1],
             [1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # 4
            [[0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0]]
            ,  # 5
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 1],
             [1, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]]
            ,  # 6
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
            ,  # 7
            [[0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0]]
            ,  # 8
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
            ,  # 9
            [[0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0]]
            ])
