from pysat.formula import IDPool, Atom, And, Or, Equals
from pysat.solvers import Solver
import numpy as np


class RectSolver:
    def __init__(self, shape: np.ndarray, k_rect: int):
        self.shape = shape
        self.n, self.m = shape.shape
        self.k_rect = k_rect
        self.vpool = IDPool()

    def __e(self, i, j): return Atom(self.vpool.id('e{},{}'.format(i, j)))

    def __s(self, k, i, j): return Atom(
        self.vpool.id('s{},{},{}'.format(k, i, j)))

    def __l(self, k, i): return Atom(self.vpool.id('l{},{}'.format(k, i)))

    def __r(self, k, i): return Atom(self.vpool.id('r{},{}'.format(k, i)))

    def __t(self, k, j): return Atom(self.vpool.id('t{},{}'.format(k, j)))

    def __c(self, k, j): return Atom(self.vpool.id('c{},{}'.format(k, j)))

    def __rect_constraints(self, k):
        constraints = []
        constraints.append(Equals(self.__l(k, 0), self.__r(k, 0)))
        for i in range(1, self.n):
            constraints.append(self.__l(k, i) >> And(
                self.__r(k, i), ~self.__r(k, i-1)))  # >> vol dir implicació
            constraints.append(
                And(self.__r(k, i), ~self.__l(k, i)) >> self.__r(k, i-1))

        constraints.append(Equals(self.__t(k, 0), self.__c(k, 0)))
        for j in range(1, self.m):
            constraints.append(self.__t(k, j) >> And(
                self.__c(k, j), ~self.__c(k, j-1)))
            constraints.append(
                And(self.__c(k, j), ~self.__t(k, j)) >> self.__c(k, j-1))

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.k_rect):
                    constraints.append(
                        Equals(self.__s(k, i, j), And(self.__r(k, i), self.__c(k, j))))

        return constraints

    def __err_constraints(self):
        constraints = []
        for i in range(self.n):
            for j in range(self.m):
                if self.shape[i][j] != 0:
                    # Si seleccionem la cel·la però no estava en la regió original compta com un error
                    constraints.append(
                        Equals(self.__e(i, j), ~Or(*[self.__s(k, i, j) for k in range(self.k_rect)])))
                else:
                    # Si no seleccionem la cel·la que estava en la regió original compta com un error
                    constraints.append(
                        Equals(self.__e(i, j), Or(*[self.__s(k, i, j) for k in range(self.k_rect)])))

        return constraints

    def solve(self):
        constraints = []
        for k in range(self.k_rect):
            constraints.append(self.__rect_constraints(k))
        err_constraints = self.__err_constraints()
        formula = And(*[c for c in err_constraints], *
                      [And(*[c for c in constraints[k]]) for k in range(self.k_rect)])

        satisfiable = True
        bound = self.n * self.m
        last_model = None
        while satisfiable:
            with Solver(name='gc4', bootstrap_with=formula) as solver:
                if not solver.solve():
                    raise Exception(
                        "Error encountered with solver. Perhaps shape is too big?")

                for k in range(self.k_rect):
                    solver.add_atmost(
                        # AMO(li)
                        lits=[self.__l(k, i).name for i in range(self.n)], k=1)
                    solver.add_atmost(
                        # AMO(tj)
                        lits=[self.__t(k, j).name for j in range(self.m)], k=1)

                solver.add_atmost(lits=[self.__e(i, j).name for i in range(self.n)
                                  for j in range(self.m)], k=bound)
                if solver.solve():
                    last_model = solver.get_model()
                    new_bound = 0
                    for i in range(self.n):
                        for j in range(self.m):
                            if self.__e(i, j).name in solver.get_model():
                                new_bound += 1
                    bound = new_bound - 1
                else:
                    satisfiable = False

        rectangles = [np.zeros((self.n, self.m)) for k in range(self.k_rect)]
        if last_model:
            for i in range(self.n):
                for j in range(self.m):
                    for k in range(self.k_rect):
                        if self.__s(k, i, j).name in last_model:
                            rectangles[k][i][j] = 1

        return rectangles
