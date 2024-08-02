from rectsolver import RectSolver
import numpy as np

shape = np.array([[0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 1]])
k_rect = 3

r = RectSolver(shape, k_rect)
for rect in r.solve():
    print(rect)

