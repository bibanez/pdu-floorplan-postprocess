import numpy as np
from rectsolver import RectSolver

def main():
    # Definim la forma que volem cobrir amb un rectangle
    shape = np.array([[0, 0, 0, 1, 1, 0],
                      [0, 1, 1, 1, 1, 0],
                      [1, 1, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 1]])
    #shape = np.array([[1,1,1,0],[1,1,0,0],[1,0,0,0]])
    #shape = np.array([[1,1,0],[1,1,0],[0,0,0]])
    
    # Use RectSolver for each region
    k_rect = 3  # You may want to adjust this for each region
    r = RectSolver(shape, k_rect)
    for rect in r.solve():
        print(rect)
        print("\n" + "="*20 + "\n")

if __name__ == "__main__":
    main()