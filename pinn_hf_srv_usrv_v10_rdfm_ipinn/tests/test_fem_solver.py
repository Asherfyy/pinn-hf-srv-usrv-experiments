from __future__ import annotations

import torch

from src.fem_solver import conjugate_gradient


def test_conjugate_gradient_solves_small_spd_system() -> None:
    indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    values = torch.tensor([4.0, 1.0, 1.0, 3.0], dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        matrix = torch.sparse_coo_tensor(indices, values, (2, 2), check_invariants=False).coalesce()
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float32)
    diagonal = torch.tensor([4.0, 3.0], dtype=torch.float32)

    result = conjugate_gradient(matrix, rhs, diagonal, tolerance=1.0e-8, max_iterations=20)

    expected = torch.linalg.solve(matrix.to_dense(), rhs)
    assert result.relative_residual < 1.0e-6
    assert torch.allclose(result.solution, expected, atol=1.0e-6)
