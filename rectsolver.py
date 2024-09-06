import numpy as np
from pysat.formula import And, Atom, Equals, IDPool, Or
from pysat.solvers import Solver


class RectSolver:
    def __init__(self, shape: np.ndarray, k_rect: int):
        self.shape = shape
        self.n, self.m = shape.shape
        self.k_rect = k_rect
        self.vpool = IDPool()

    # Error (i, j)
    def __e(self, i, j): return Atom(self.vpool.id('e{},{}'.format(i, j)))

    # Selected by rectangle k (i, j)
    def __s(self, k, i, j): return Atom(
        self.vpool.id('s{},{},{}'.format(k, i, j)))

    # Left start in row i by rectangle k
    def __l(self, k, i): return Atom(self.vpool.id('l{},{}'.format(k, i)))

    # Row i selected by rectangle k
    def __r(self, k, i): return Atom(self.vpool.id('r{},{}'.format(k, i)))

    # Top start in column j by rectangle k
    def __t(self, k, j): return Atom(self.vpool.id('t{},{}'.format(k, j)))

    # Column j selected by rectangle k
    def __c(self, k, j): return Atom(self.vpool.id('c{},{}'.format(k, j)))

    # Rectangle constraints, not including 'at most one' constraints
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
                        Equals(self.__s(k, i, j),
                               And(self.__r(k, i), self.__c(k, j))))

        return constraints

    def __err_constraints(self):
        constraints = []
        for i in range(self.n):
            for j in range(self.m):
                # We compute an error if the cell is selected but it is not
                # part of the original shape, or if the cell is not selected
                # and it is part of the shape.
                if self.shape[i][j] != 0:
                    constraints.append(
                        Equals(self.__e(i, j),
                               ~Or(*[self.__s(k, i, j)
                                   for k in range(self.k_rect)])))
                else:
                    constraints.append(
                        Equals(self.__e(i, j),
                               Or(*[self.__s(k, i, j)
                                  for k in range(self.k_rect)])))

        return constraints

    def __add_amo(self, solver: Solver, bound: int, largest_rect: int):
        for k in range(self.k_rect):
            solver.add_atmost(
                # AMO(li)
                lits=[self.__l(k, i).name for i in range(self.n)], k=1)
            solver.add_atmost(
                # AMO(tj)
                lits=[self.__t(k, j).name for j in range(self.m)], k=1)
        for i in range(self.n):
            for j in range(self.m):
                solver.add_atmost(
                    lits=[self.__s(k, i, j).name
                          for k in range(self.k_rect)], k=1)
        solver.add_atmost(lits=[self.__e(i, j).name
                          for i in range(self.n)
                          for j in range(self.m)], k=bound)
        if largest_rect > 0:
            solver.add_atmost(lits=[-self.__s(0, i, j).name
                              for i in range(self.n)
                              for j in range(self.m)],
                              k=self.n * self.m - largest_rect)

    def __get_rect(self, model):
        rectangles = [np.zeros((self.n, self.m), dtype=int)
                      for k in range(self.k_rect)]

        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.k_rect):
                    if self.__s(k, i, j).name in model:
                        rectangles[k][i][j] = 1

        return rectangles

    def __find_error_lower_bound(self, formula):
        # Find lower bound for error
        satisfiable = True
        # Upper bound of the error (every cell)
        bound = self.n * self.m
        # Last model that was solved
        last_model = None
        # Lower bound for the error
        last_bound = None

        # Area of the largest rectangle. This will serve as a lower bound in
        # the next step of the algorithm.
        largest_rect = 0

        while satisfiable:
            with Solver(name='gc4', bootstrap_with=formula) as solver:
                if not solver.solve():
                    raise Exception(
                        "Error encountered with solver. \
                        Perhaps shape is too big?")

                # Count the area of all rectangles
                rect_area = np.zeros(self.k_rect, dtype=int)

                self.__add_amo(solver, bound, 0)

                if solver.solve():
                    last_model = solver.get_model()
                    last_bound = bound

                    # We subtract 1 from the new bound
                    bound = -1
                    for i in range(self.n):
                        for j in range(self.m):
                            if self.__e(i, j).name in solver.get_model():
                                bound += 1
                            for k in range(self.k_rect):
                                if self.__s(k, i, j).name in last_model:
                                    rect_area[k] += 1
                    largest_rect = np.max(rect_area)
                    rect_area = np.zeros(self.k_rect, dtype=int)
                else:
                    satisfiable = False

        return last_bound, largest_rect

    def __find_solution_largest_rectangle(self, formula, last_bound,
                                          largest_rect):
        # Optimize for largest single rectangle with the new calculated lower
        # bound for error
        last_model = None
        satisfiable = True
        while satisfiable:
            with Solver(name='gc4', bootstrap_with=formula) as solver:
                if not solver.solve():
                    raise Exception(
                        "Error encountered with solver. \
                        Perhaps shape is too big?")

                self.__add_amo(solver, last_bound, largest_rect)

                rect_area = np.zeros(self.k_rect, dtype=int)

                if solver.solve():
                    last_model = solver.get_model()
                    for i in range(self.n):
                        for j in range(self.m):
                            for k in range(self.k_rect):
                                if self.__s(k, i, j).name in last_model:
                                    rect_area[k] += 1

                    largest_rect = np.max(rect_area) + 1
                    rect_area = np.zeros(self.k_rect, dtype=int)
                else:
                    satisfiable = False

        return last_model

    def solve(self):
        constraints = []
        for k in range(self.k_rect):
            constraints.append(self.__rect_constraints(k))
        err_constraints = self.__err_constraints()

        # The reason we do this is to be able to perform an And operation of
        # all conditions at once (which makes the solution cleaner in my
        # opinion). There are some other And constraints nested in because
        # of a bug where I found that too many arguments in a single And
        # operation made the SAT solver crash.
        formula = And(*[c for c in err_constraints],
                      *[And(*[c for c in constraints[k]])
                          for k in range(self.k_rect)])

        last_bound, largest_rect = self.__find_error_lower_bound(formula)

        # If no solutions were found, return empty
        if last_bound is None:
            return None

        last_model = self.__find_solution_largest_rectangle(formula,
                                                            last_bound,
                                                            largest_rect)

        return self.__get_rect(last_model)
