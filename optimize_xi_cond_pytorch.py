"""
Minimize the spectral condition number κ(ξ A) using PyTorch autograd, with ξ
constrained to the Kronecker-row structure from build_xi_design_matrix:

    ξ[r, :] = kron(f_i, g_j), with i = r % c, j = r // c

where F = [f_1,...,f_c] and G = [g_1,...,g_c] are trainable matrices of shape
(c, L), L = 2N-2.

Shapes (matching your EIT notebook):
  L = 2*N - 2
  A : (L^2, N^2)   — rows indexed like (source, receiver) pairs
  ξ : (c^2, L^2)
  ξ A : (c^2, N^2)

Loss: κ(M) = σ_max(M) / σ_min(M) + ε  with M = ξ A, using torch.linalg.svdvals.
Because κ is scale-invariant, ξ is Frobenius-normalized inside the forward pass.

Requires: pip install torch
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# allow `from build_xi_design_matrix import build_xi` when run from elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parent))


def L_from_N(N: int) -> int:
    return 2 * N - 2


def spectral_cond_torch(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable (almost everywhere) spectral condition number σ_max / σ_min."""
    # min(m, n) singular values for m × n matrix
    s = torch.linalg.svdvals(M)
    return s[0] / (s[-1] + eps)


def frob_normalize(xi: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    n = torch.linalg.norm(xi.reshape(-1))
    return xi * (target / (n + 1e-30))


def build_xi_from_FG_torch(F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Build ξ with the exact row ordering:
      [kron(f0,g0), kron(f1,g0), ..., kron(f_{c-1},g0),
       kron(f0,g1), ..., kron(f_{c-1},g_{c-1})]
    where F,G are (c, L). Output ξ is (c^2, L^2).
    """
    c, L = F.shape
    if G.shape != (c, L):
        raise ValueError(f"G must have shape {(c, L)}, got {tuple(G.shape)}")

    # Same ordering as build_xi_design_matrix.py: row r = j*c + i is kron(F[i], G[j]).
    return (
        (F[:, None, :, None] * G[None, :, None, :])
        .permute(1, 0, 2, 3)
        .reshape(c * c, L * L)
    )


def optimize_xi_cond(
    A_np: np.ndarray,
    c: int,
    N: int,
    *,
    steps: int = 500,
    lr: float = 0.05,
    seed: int = 0,
    init_from_kron: bool = False,
    F_np: Optional[np.ndarray] = None,
    G_np: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    save_prefix: Optional[str] = None,
    return_f_g: bool = False,
):
    """
    Parameters
    ----------
    A_np : (L^2, N^2) with L = 2N-2
        Your **FEM-derived sensitivity** matrix from the notebook (the dense
        `A` built from grad u^i · grad v^j on the grid, shape (n1*n2, N*N) with
        n1=n2=2N-2). This function does **not** load FEM data; pass that array in.
    c : number of f / g vectors → ξ is (c^2, L^2)
    init_from_kron :
        If True, initialize trainable F,G from F_np,G_np.
        If False, initialize trainable F,G randomly.
    F_np, G_np :
        Optional initial arrays of shape (c, L).
    save_prefix :
        If provided, save trained arrays to:
          <save_prefix>_F.npy, <save_prefix>_G.npy, <save_prefix>_xi.npy
        and print shape checks after saving.
    return_f_g :
        If True, return (xi_opt, history, F_opt, G_opt).
        If False, return (xi_opt, history) for backward compatibility.
    """
    L = L_from_N(N)
    n_rows_A = L * L
    n_cols_A = N * N
    A_np = np.asarray(A_np, dtype=np.float64)
    if A_np.shape != (n_rows_A, n_cols_A):
        raise ValueError(
            f"A must have shape ({n_rows_A}, {n_cols_A}) for N={N} (L=2N-2={L}), got {A_np.shape}"
        )

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    A = torch.tensor(A_np, dtype=torch.float64, device=dev)

    # Trainable variables are F and G (structure-preserving parameterization of ξ).
    F = torch.empty(c, L, dtype=torch.float64, device=dev, requires_grad=True)
    G = torch.empty(c, L, dtype=torch.float64, device=dev, requires_grad=True)

    if init_from_kron:
        if F_np is None or G_np is None:
            raise ValueError("init_from_kron=True requires both F_np and G_np")
        F_init = np.asarray(F_np, dtype=np.float64)
        G_init = np.asarray(G_np, dtype=np.float64)
        if F_init.shape != (c, L) or G_init.shape != (c, L):
            raise ValueError(
                f"F_np and G_np must each have shape ({c}, {L}); "
                f"got {F_init.shape} and {G_init.shape}"
            )
        with torch.no_grad():
            F.copy_(torch.tensor(F_init, dtype=torch.float64, device=dev))
            G.copy_(torch.tensor(G_init, dtype=torch.float64, device=dev))
    else:
        # Xavier-like random init for row vectors
        torch.nn.init.normal_(F, mean=0.0, std=1.0 / np.sqrt(L))
        torch.nn.init.normal_(G, mean=0.0, std=1.0 / np.sqrt(L))

    opt = torch.optim.Adam([F, G], lr=lr)
    history: list[float] = []

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        xi = build_xi_from_FG_torch(F, G)
        xi = frob_normalize(xi)
        M = xi @ A
        loss = spectral_cond_torch(M)
        k_after = float(loss.detach().cpu())
        loss.backward()
        opt.step()

        history.append(k_after)

        if t % 50 == 0 or t == steps - 1:
            print(f"step {t:4d}  cond(ξA) ≈ {k_after:.6e}")

    xi_final = frob_normalize(build_xi_from_FG_torch(F, G))
    xi_final_np = xi_final.detach().cpu().numpy()
    F_final_np = F.detach().cpu().numpy()
    G_final_np = G.detach().cpu().numpy()

    if save_prefix is not None:
        f_path = f"{save_prefix}_F.npy"
        g_path = f"{save_prefix}_G.npy"
        xi_path = f"{save_prefix}_xi.npy"

        np.save(f_path, F_final_np)
        np.save(g_path, G_final_np)
        np.save(xi_path, xi_final_np)

        # shape checks after saving
        F_chk = np.load(f_path, mmap_mode="r")
        G_chk = np.load(g_path, mmap_mode="r")
        xi_chk = np.load(xi_path, mmap_mode="r")
        print("Saved F to", f_path, "shape:", F_chk.shape)
        print("Saved G to", g_path, "shape:", G_chk.shape)
        print("Saved xi to", xi_path, "shape:", xi_chk.shape)

    if return_f_g:
        return xi_final_np, history, F_final_np, G_final_np
    return xi_final_np, history


def demo():
    """
    **Standalone test only:** builds a random `A`, not your notebook FEM `A`.
    For real EIT runs, call `optimize_xi_cond(your_A, ...)` from the notebook
    after you build the sensitivity matrix `A`.
    """
    N = 8
    c = 4
    L = L_from_N(N)
    rng = np.random.default_rng(1)
    # Tall-ish random A for a nontrivial product
    A = rng.standard_normal((L * L, N * N))
    # Mild column scaling so cond(ξA) is not trivial
    for j in range(N * N):
        A[:, j] *= 0.5 + (j % 7) * 0.1

    print(f"N={N}, L=2N-2={L}, A shape {A.shape}, c={c}, ξ shape ({c*c}, {L*L})")
    xi_opt, hist = optimize_xi_cond(A, c=c, N=N, steps=400, lr=0.08, seed=2)

    M = xi_opt @ A
    k_np = np.linalg.cond(M)
    print("numpy cond(ξ_opt A):", k_np)
    print("first 5 loss values:", hist[:5])
    print("last 5 loss values:", hist[-5:])


if __name__ == "__main__":
    demo()
