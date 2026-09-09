import numpy as np
import torch

def create_samples():
    # numpy
    v_np = np.array([1, 2, 3])
    M_np = np.array([[1,2], [3,4],[5,6]])

    # pytorch
    v_pt = torch.tensor([1,2,3], dtype=torch.float32)
    M_pt = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    return (v_np, M_np), (v_pt, M_pt)

def dot_product(a, b, backend='numpy'):
    if backend == 'numpy':
        return np.dot(a, b) # a @ b
    elif backend == 'pytorch':
        return torch.dot(a, b) # a @ b

def matrix_multiplication(A, B, backend='numpy'):
    if backend == 'numpy':
        return np.matmul(A, B) # A @ B
    elif backend == 'pytorch':
        return torch.matmul(A, B) # A @ B

def transpose_matrix(M, backend='numpy'):
    if backend == 'numpy':
        return M.T
    elif backend == 'pytorch':
        return M.T

def L2_norm(v, backend='numpy'):
    if backend == 'numpy':
        return np.linalg.norm(v)
    elif backend == 'pytorch':
        return torch.norm(v)

def invert_matrix(A, backend='numpy'):
    if backend == 'numpy':
        return np.linalg.inv(A)
    elif backend == 'pytorch':
        return torch.inverse(A)

def solve_linear_system(A, b, backend='numpy'):
    if backend == 'numpy':
        return np.linalg.solve(A, b)
    elif backend == 'pytorch':
        return torch.linalg.solve(A, b)

