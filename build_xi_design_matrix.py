"""
Design matrix ξ of shape (c^2, L^2): each row is the Kronecker product (as a row
vector) of f_i and g_j, with ordering

  row 0      : f_0 ⊗ g_0
  row 1      : f_1 ⊗ g_0
  ...
  row c-1    : f_{c-1} ⊗ g_0
  row c      : f_0 ⊗ g_1
  ...
  row c^2-1  : f_{c-1} ⊗ g_{c-1}

F, G are (c, L) with row i equal to f_i and g_i (row vectors in R^L).

For your EIT setup, take L = 2*N - 2 so L^2 = (2N-2)^2 matches the row count of A.
"""

from __future__ import annotations

import numpy as np


def build_xi(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    F : (c, L) — row i is f_i (as a row vector in R^L)
    G : (c, L) — row i is g_i

    Returns
    -------
    xi : (c^2, L^2) — each row is np.kron(f_i, g_j) with the ordering above.
    """
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    if F.ndim != 2 or G.ndim != 2:
        raise ValueError("F and G must be 2D arrays")
    c_f, L_f = F.shape
    c_g, L_g = G.shape
    if c_f != c_g:
        raise ValueError("F and G must have the same number of rows (c)")
    if L_f != L_g:
        raise ValueError("F and G must have the same number of columns (L)")
    c, L = c_f, L_f

    xi = np.zeros((c * c, L * L), dtype=np.float64)
    for r in range(c * c):
        i = r % c   # which f
        j = r // c  # which g
        # np.kron for row vectors (1, L) gives (1, L^2)
        row = np.kron(F[i : i + 1, :], G[j : j + 1, :])
        xi[r, :] = row.ravel()
    return xi


def example():
    np.random.seed(0)
    c = 3
    L = 4  # e.g. L = 2*N-2 when matching A's (2N-2)^2 rows

    F = np.random.randn(c, L)
    G = np.random.randn(c, L)

    xi = build_xi(F, G)
    print("c =", c, "  L =", L)
    print("xi shape:", xi.shape, "  expected:", (c * c, L * L))

    # Spot-check first few rows vs manual kron
    for r in range(min(5, c * c)):
        i, j = r % c, r // c
        ref = np.kron(F[i : i + 1, :], G[j : j + 1, :]).ravel()
        err = np.linalg.norm(xi[r] - ref)
        print(f"row {r} (f_{i}, g_{j}): kron check err = {err:.2e}")

    # Optional: verify ξ @ A works when A has (L^2, n) — toy A
    n = 5
    A = np.random.randn(L * L, n)
    prod = xi @ A
    print("xi @ A shape:", prod.shape, "  expected:", (c * c, n))


if __name__ == "__main__":
    example()
