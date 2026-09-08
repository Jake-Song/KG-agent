"""Fixed, stdlib-only polynomial regression benchmark; no agent-authored code."""
import math
import random


def dataset(seed: int, split: str):
    offsets = {"train": 0, "validation": 1, "test": 2}
    rng = random.Random(seed * 3 + offsets[split])
    count = 80 if split == "train" else 40
    return [(x, 0.5 + 1.5*x - 2*x*x + rng.gauss(0, 0.1))
            for x in (rng.uniform(-1, 1) for _ in range(count))]


def fit(rows, degree, penalty):
    """Solve mean squared error + ridge * squared non-intercept coefficients."""
    size = degree + 1
    features = [[x**i for i in range(size)] for x, _ in rows]
    matrix = [[sum(v[i]*v[j] for v in features) / len(rows)
               + (penalty if i == j and i > 0 else 0)
               for j in range(size)]
              + [sum(v[i]*y for v, (_, y) in zip(features, rows)) / len(rows)]
              for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(matrix[r][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        if abs(divisor) < 1e-14:
            raise ArithmeticError("Singular regression system")
        matrix[col] = [v / divisor for v in matrix[col]]
        for row in range(size):
            if row != col:
                factor = matrix[row][col]
                matrix[row] = [a - factor*b for a, b in zip(matrix[row], matrix[col])]
    coefficients = [row[-1] for row in matrix]
    if not all(math.isfinite(v) for v in coefficients):
        raise ArithmeticError("Non-finite model")
    return coefficients


def mse(rows, coefficients):
    return sum((sum(c*x**i for i, c in enumerate(coefficients)) - y)**2
               for x, y in rows) / len(rows)
