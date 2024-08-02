from pysat.formula import IDPool, Atom, And, Or, Equals
from pysat.solvers import Solver
import numpy as np

# Definim la forma que volem cobrir amb un rectangle
shape = np.array([[0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 1]])
# shape = np.array([[1,1,1,0],[1,1,0,0],[1,0,0,0]])
n = shape.shape[0]
m = shape.shape[1]

# De moment només treballarem amb k=1 rectangles
vpool = IDPool()  # IDPool() controla el canvi de nom <-> literal (nombre >= 1)
def e(i, j): return Atom(vpool.id('e{},{}'.format(i, j)))
def s(k, i, j): return Atom(vpool.id('s{},{},{}'.format(k, i, j)))
def l(k, i): return Atom(vpool.id('l{},{}'.format(k, i)))
def r(k, i): return Atom(vpool.id('r{},{}'.format(k, i)))
def t(k, j): return Atom(vpool.id('t{},{}'.format(k, j)))
def c(k, j): return Atom(vpool.id('c{},{}'.format(k, j)))


# Definim totes les fórmules que després hauran d'anar juntes amb un And
k_rect = 3
constraints = [[] for k in range(k_rect)]
for k in range(k_rect):
    # Restriccions files
    constraints[k].append(Equals(l(k, 0), r(k, 0)))
    for i in range(1, n):
        constraints[k].append(l(k, i) >> And(
            r(k, i), ~r(k, i-1)))  # >> vol dir implicació
        constraints[k].append(And(r(k, i), ~l(k, i)) >> r(k, i-1))

    # Restriccions columnes
    constraints[k].append(Equals(t(k, 0), c(k, 0)))
    for j in range(1, m):
        constraints[k].append(t(k, j) >> And(c(k, j), ~c(k, j-1)))
        constraints[k].append(And(c(k, j), ~t(k, j)) >> c(k, j-1))

err_constraints = []
for i in range(n):
    for j in range(m):
        # Variable de seleccionar
        for k in range(k_rect):
            constraints[k].append(Equals(s(k, i, j), And(r(k, i), c(k, j))))
        if shape[i][j] != 0:
            # Si seleccionem la cel·la però no estava en la regió original compta com un error
            err_constraints.append(
                Equals(e(i, j), ~Or(*[s(k, i, j) for k in range(k_rect)])))
        else:
            # Si no seleccionem la cel·la que estava en la regió original compta com un error
            err_constraints.append(
                Equals(e(i, j), Or(*[s(k, i, j) for k in range(k_rect)])))

formula = And(*[c for c in err_constraints], *
              [And(*[c for c in constraints[k]]) for k in range(k_rect)])

satisfiable = True
bound = n*m
last_model = None
while satisfiable:
    with Solver(name='gc4', bootstrap_with=formula) as solver:
        if not solver.solve():
            print("unsatisfiability confirmed")
            break
        for k in range(k_rect):
            solver.add_atmost(
                lits=[l(k, i).name for i in range(n)], k=1)  # AMO(li)
            solver.add_atmost(
                lits=[t(k, j).name for j in range(m)], k=1)  # AMO(tj)
        solver.add_atmost(lits=[e(i, j).name for i in range(n)
                          for j in range(m)], k=bound)
        if solver.solve():
            print("satisfiable")
            last_model = solver.get_model()
            new_bound = 0
            for i in range(n):
                for j in range(m):
                    if e(i, j).name in solver.get_model():
                        new_bound += 1
            print("New bound", new_bound)
            bound = new_bound - 1
        else:
            satisfiable = False
            print("unsatisfiable")

result = np.zeros((n, m))
rectangles = [np.zeros((n, m)) for k in range(k_rect)]
if last_model:
    for i in range(n):
        for j in range(m):
            for k in range(k_rect):
                if s(k, i, j).name in last_model:
                    result[i][j] = 1
                    rectangles[k][i][j] = 1

print(result)
for k in range(k_rect):
    print("Rectangle", k)
    print(rectangles[k])
