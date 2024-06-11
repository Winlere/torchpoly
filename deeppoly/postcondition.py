import torch
import torch.nn as nn
import deeppoly.base as base
import deeppoly.linear

class ArgumentedPostcond(deeppoly.linear.ArgumentedLinear):
        def __init__(self, A, b):
                """Accept a postcondition and construct a postcondition layer. (The nature is a linear)
                A postcondition are linear constraint(s) Ax + b >= 0. 
                Args:
                    A (array-like): The matrix A. Each row represents a constraint.
                    b (array-like): The vector b.
                """
                temp = nn.Linear(A.shape[1], A.shape[0])
                temp.train(False)
                temp.weight = nn.Parameter(torch.tensor(A, dtype=torch.float32))
                temp.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32))
                super().__init__(temp)

ALL = [ArgumentedPostcond]