"""
Lennard-Jones fluid MD with CuPy GPU acceleration.
Coexistence curve via slab geometry.
Runs N = 1000, 2000, 4000 for both dim=2 and dim=3.

Usage:
  python md_lj.py          # run everything
  python md_lj.py 2 1000   # dim=2, N=1000 only
  python md_lj.py 3 4000   # dim=3, N=4000 only
"""
import numpy as np
import sys

try:
    import cupy as cp
    _GPU = True
    cp.zeros(1)  # warm up CUDA context
    xp = cp
except Exception:
    cp = None
    _GPU = False
    xp = np

RCUT    = 2.5
_r6c    = RCUT ** -6
V_SHIFT = 4.0 * (_r6c ** 2 - _r6c)   # truncated-shifted LJ


# ── batched force calculation ─────────────────────────────────────────────────

def lj_forces_batch(r, box):
    """
    r   : (B, N, dim) — positions for B independent simulations
    box : (dim,)
    Returns F (B, N, dim) and pe (B,).
    """
    B, N, dim = r.shape

    dr  = r[:, :, None, :] - r[:, None, :, :]   # (B, N, N, dim)
    dr -= box * xp.round(dr / box)               # minimum image

    r2  = (dr * dr).sum(-1)                      # (B, N, N)

    # zero out self-interaction on diagonal
    idx = xp.arange(N)
    r2[:, idx, idx] = xp.inf

    mask = r2 < RCUT * RCUT
    r2i  = xp.where(mask, 1.0 / r2, 0.0)
    r6i  = r2i ** 3
    r12i = r6i ** 2

    fij = xp.where(mask, 24.0 * r2i * (2.0 * r12i - r6i), 0.0)   # (B,N,N)
    F   = (fij[..., None] * dr).sum(2)                             # (B,N,dim)

    pe  = 0.5 * xp.where(mask, 4.0 * (r12i - r6i) - V_SHIFT, 0.0).sum(axis=(1, 2))  # (B,)
    return F, pe


# ── velocity rescaling thermostat (batched) ───────────────────────────────────

def rescale_batch(v, T_target_cp):
    """v: (B,N,dim), T_target_cp: (B,1,1)"""
    T_now = (v * v).sum(axis=(1, 2), keepdims=True) / (v.shape[1] * v.shape[2])
    v *= xp.sqrt(T_target_cp / xp.maximum(T_now, 1e-12))
    return v


# ── initialisation ────────────────────────────────────────────────────────────

def _grid_in_box(N, box, dim):
    """Deterministic rectangular grid inside box. CPU (numpy)."""
    if dim == 2:
        Lx, Ly = box
        nx = max(1, round(np.sqrt(N * Lx / Ly)))
        ny = int(np.ceil(N / nx))
        xs = (np.arange(nx) + 0.5) * Lx / nx
        ys = (np.arange(ny) + 0.5) * Ly / ny
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        pos = np.column_stack([xx.ravel(), yy.ravel()])
    else:
        Lx, Ly, Lz = box
        ny = max(1, round((N * Ly * Lz / Lx) ** (1 / 3)))
        nz = max(1, round((N * Ly * Lz / Lx) ** (1 / 3)))
        nx = int(np.ceil(N / (ny * nz)))
        xs = (np.arange(nx) + 0.5) * Lx / nx
        ys = (np.arange(ny) + 0.5) * Ly / ny
        zs = (np.arange(nz) + 0.5) * Lz / nz
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        pos = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return pos[:N]


def _slab_single(N, box, dim, rng, frac_liquid=0.85):
    """Center-slab IC: all N particles in center half, vacuum on both sides.

    Standard approach for LJ coexistence: avoids over-dense gas init that
    causes premature condensation. Gas forms naturally during equilibration.
    """
    Lx = box[0]
    box_liq = box.copy()
    box_liq[0] = Lx / 2.0          # liquid block occupies center half in x
    r = _grid_in_box(N, box_liq, dim)
    r += rng.uniform(-0.02, 0.02, r.shape)   # small jitter to break symmetry
    r[:, 0] += Lx / 4.0            # shift: x ∈ [Lx/4, 3Lx/4]
    r[:, 0] %= Lx
    return r


def init_batch(B, N, box_np, dim, T_batch_np, seed=42):
    """
    Returns r (B,N,dim) and v (B,N,dim) on GPU (or CPU if no GPU).
    Each sim gets independent slab IC + MB velocities at its target T.
    """
    rng = np.random.default_rng(seed)
    r_np = np.stack([_slab_single(N, box_np, dim, rng) for _ in range(B)])
    v_np = rng.standard_normal((B, N, dim))
    v_np -= v_np.mean(axis=1, keepdims=True)  # zero net momentum per sim

    # scale velocities to target T
    T_now = (v_np ** 2).sum(axis=(1, 2)) / (N * dim)  # (B,)
    scale = np.sqrt(T_batch_np / np.maximum(T_now, 1e-12))
    v_np *= scale[:, None, None]

    return xp.asarray(r_np), xp.asarray(v_np)


# ── density-profile extraction ────────────────────────────────────────────────

def extract_rho(rho_x, frac=0.2):
    s = np.sort(rho_x)
    k = max(1, round(frac * len(s)))
    return s[-k:].mean(), s[:k].mean()


# ── main simulation loop ──────────────────────────────────────────────────────

def run_md(dim, N, T_arr, rho_c=None,
           n_eq=25000, n_meas=25000, dt=0.005,
           thermo_every=10, n_bins=80, batch_size=4, seed=42):

    if rho_c is None:
        rho_c = 0.35 if dim == 2 else 0.48   # 3D: slab init density 2*rho_c=0.96 > rho_l~0.75

    V = N / rho_c
    if dim == 2:
        Ly      = np.sqrt(V / 4.0)
        box_np  = np.array([4.0 * Ly, Ly])
    else:
        Lyz     = (V / 4.0) ** (1.0 / 3.0)
        box_np  = np.array([4.0 * Lyz, Lyz, Lyz])

    box = xp.asarray(box_np)
    bin_vol = (box_np[0] / n_bins) * np.prod(box_np[1:])

    n_T       = len(T_arr)
    rho_l_arr = np.zeros(n_T)
    rho_g_arr = np.zeros(n_T)
    n_batches = -(-n_T // batch_size)

    for bidx, b0 in enumerate(range(0, n_T, batch_size)):
        b1       = min(b0 + batch_size, n_T)
        B        = b1 - b0
        T_batch  = T_arr[b0:b1]
        T_gpu    = xp.asarray(T_batch)[:, None, None]   # (B,1,1)

        print(f"  dim={dim}  N={N}  batch {bidx+1}/{n_batches}"
              f"  T={T_batch[0]:.3f}–{T_batch[-1]:.3f}", flush=True)

        r, v = init_batch(B, N, box_np, dim, T_batch, seed=seed + b0)
        F, _ = lj_forces_batch(r, box)

        # equilibration
        for step in range(n_eq):
            v  += 0.5 * dt * F
            r   = (r + dt * v) % box
            F, _ = lj_forces_batch(r, box)
            v  += 0.5 * dt * F
            if (step + 1) % thermo_every == 0:
                v = rescale_batch(v, T_gpu)

        # production — accumulate density profile
        rho_x = xp.zeros((B, n_bins))
        for step in range(n_meas):
            v  += 0.5 * dt * F
            r   = (r + dt * v) % box
            F, _ = lj_forces_batch(r, box)
            v  += 0.5 * dt * F
            if (step + 1) % thermo_every == 0:
                v = rescale_batch(v, T_gpu)

            # batched bincount: flatten (B,N) indices into (B*n_bins,) space
            x_frac = r[:, :, 0] / box[0]                                 # (B,N)
            bin_i  = (x_frac * n_bins).astype(xp.int32).clip(0, n_bins-1)
            flat_i = bin_i + xp.arange(B, dtype=xp.int32)[:, None] * n_bins
            counts = xp.bincount(flat_i.ravel(), minlength=B * n_bins)
            rho_x += counts.reshape(B, n_bins)

        rho_x_np = (cp.asnumpy(rho_x) if _GPU else rho_x) / (n_meas * bin_vol)

        for b in range(B):
            rl, rg = extract_rho(rho_x_np[b])
            rho_l_arr[b0 + b] = rl
            rho_g_arr[b0 + b] = rg
            print(f"    T={T_batch[b]:.4f}  rho_l={rl:.4f}  rho_g={rg:.4f}", flush=True)

    return dict(T=T_arr, rho_l=rho_l_arr, rho_g=rho_g_arr, box=box_np, dim=dim, N=N)


# ── entry point ───────────────────────────────────────────────────────────────

def _batch_size(N):
    """Heuristic batch size by N to stay under ~8 GB VRAM for the (B,N,N,dim) tensor."""
    if N <= 1000: return 8
    if N <= 2000: return 4
    return 2


def run_all(dim, N_list, T_arr_2d, T_arr_3d):
    for N in N_list:
        T_arr = T_arr_2d if dim == 2 else T_arr_3d
        tag   = f"md{dim}d_N{N}"
        print(f"\n=== MD  dim={dim}  N={N} ===")
        data  = run_md(dim, N, T_arr, batch_size=_batch_size(N))
        np.save(f"{tag}.npy", data)
        print(f"Saved {tag}.npy")


def main():
    T_2d      = np.linspace(0.36, 0.44, 24)   # coexistence curve (existing)
    T_2d_near = np.linspace(0.43, 0.46, 8)    # near T_c for beta; t=0.04-0.10
    T_3d      = np.linspace(0.78, 0.84, 8)    # 3D coexistence; rho_c now 0.48
    N_list = [1000]

    if len(sys.argv) >= 3:
        dim  = int(sys.argv[1]); N = int(sys.argv[2])
        mode = sys.argv[3] if len(sys.argv) == 4 else ""
        if dim == 2 and mode == "nearTc":
            data = run_md(dim, N, T_2d_near, batch_size=_batch_size(N))
            np.save(f"md2d_N{N}_nearTc.npy", data)
            print(f"Saved md2d_N{N}_nearTc.npy")
        else:
            T_arr = T_2d if dim == 2 else T_3d
            data  = run_md(dim, N, T_arr, batch_size=_batch_size(N))
            np.save(f"md{dim}d_N{N}.npy", data)
            print(f"Saved md{dim}d_N{N}.npy")
        return

    for dim in [2, 3]:
        run_all(dim, N_list, T_2d, T_3d)


if __name__ == '__main__':
    main()
