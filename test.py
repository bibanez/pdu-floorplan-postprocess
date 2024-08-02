from pypblib import pblib
import numpy as np
formula = pblib.VectorClauseDatabase(pblib.PBConfig(),
                        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
formula.print_formula()
