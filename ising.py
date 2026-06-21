"""
2D Ising model:
  L = 32, 64, 128, 256 — GPU checkerboard Metropolis (CuPy, T-batched)
  L = 512               — CPU Wolff cluster algorithm (numba, multiprocessing)
Saves ising_data.npy : dict{ L -> dict{T, M, M2, M4, E, E2} }
"""
import numpy as np
from multiprocessing import Pool
from numba import njit

try:
    import cupy as cp
    _GPU = True
    cp.zeros(1)  # warm up CUDA context
except Exception:
    cp = None
    _GPU = False

TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))


# ── GPU checkerboard Metropolis ───────────────────────────────────────────────

def run_metropolis_gpu(L, T_arr, n_eq=5000, n_meas=200000, batch_size=24, seed=42):
    assert _GPU, "CuPy unavailable — cannot run GPU Metropolis"
    n_T = len(T_arr)
    M  = np.zeros(n_T); M2 = np.zeros(n_T); M4 = np.zeros(n_T)
    E  = np.zeros(n_T); E2 = np.zeros(n_T)
    N_sites = L * L

    cp.random.seed(seed)

    # Checkerboard sublattice masks: (1, L, L)
    ii, jj = cp.indices((L, L))
    mA = ((ii + jj) % 2 == 0)[None]
    mB = ~mA

    n_batches = -(-n_T // batch_size)
    for bidx, b0 in enumerate(range(0, n_T, batch_size)):
        b1 = min(b0 + batch_size, n_T)
        B  = b1 - b0
        beta = cp.array(1.0 / T_arr[b0:b1], dtype=cp.float32)[:, None, None]

        spin = (cp.random.randint(0, 2, (B, L, L)) * 2 - 1).astype(cp.float32)

        def upd(mask):
            nb = (cp.roll(spin, 1, 1) + cp.roll(spin, -1, 1) +
                  cp.roll(spin, 1, 2) + cp.roll(spin, -1, 2))
            dE   = 2.0 * spin * nb
            rand = cp.random.random((B, L, L), dtype=cp.float32)
            acc  = mask & ((dE <= 0) | (rand < cp.exp(-beta * cp.maximum(dE, 0.0))))
            return cp.where(acc, -spin, spin)

        for _ in range(n_eq):
            spin = upd(mA); spin = upd(mB)

        ms = cp.zeros(B); m2s = cp.zeros(B)
        m4s = cp.zeros(B); es = cp.zeros(B); e2s = cp.zeros(B)

        for _ in range(n_meas):
            spin = upd(mA); spin = upd(mB)
            m  = spin.sum(axis=(1, 2)) / N_sites
            e  = -(spin * (cp.roll(spin, 1, 1) + cp.roll(spin, 1, 2))).sum(axis=(1, 2)) / N_sites
            am = cp.abs(m)
            ms += am; m2s += m*m; m4s += m**4; es += e; e2s += e*e

        M[b0:b1]  = cp.asnumpy(ms  / n_meas)
        M2[b0:b1] = cp.asnumpy(m2s / n_meas)
        M4[b0:b1] = cp.asnumpy(m4s / n_meas)
        E[b0:b1]  = cp.asnumpy(es  / n_meas)
        E2[b0:b1] = cp.asnumpy(e2s / n_meas)
        print(f"  L={L:4d}  batch {bidx+1}/{n_batches}  T={T_arr[b0]:.3f}–{T_arr[b1-1]:.3f}",
              flush=True)

    return dict(T=T_arr, M=M, M2=M2, M4=M4, E=E, E2=E2)


# ── CPU Wolff (numba JIT) ─────────────────────────────────────────────────────

@njit(cache=True)
def _wolff_step(spin, L, p_add):
    """One Wolff cluster flip. Returns cluster size."""
    r0 = np.random.randint(0, L)
    c0 = np.random.randint(0, L)
    s0 = spin[r0, c0]

    cap = L * L
    stk_r = np.empty(cap, np.int32);  stk_c = np.empty(cap, np.int32)
    mem_r = np.empty(cap, np.int32);  mem_c = np.empty(cap, np.int32)
    in_cl = np.zeros((L, L), np.bool_)

    stk_r[0] = r0; stk_c[0] = c0
    in_cl[r0, c0] = True
    mem_r[0] = r0; mem_c[0] = c0
    top = 1; csz = 1

    while top > 0:
        top -= 1
        row = stk_r[top]; col = stk_c[top]
        for k in range(4):
            if   k == 0: nr, nc = (row - 1) % L, col
            elif k == 1: nr, nc = (row + 1) % L, col
            elif k == 2: nr, nc = row, (col - 1) % L
            else:        nr, nc = row, (col + 1) % L
            if not in_cl[nr, nc] and spin[nr, nc] == s0:
                if np.random.random() < p_add:
                    in_cl[nr, nc] = True
                    stk_r[top] = nr; stk_c[top] = nc; top += 1
                    mem_r[csz] = nr; mem_c[csz] = nc; csz += 1

    s_new = -s0
    for k in range(csz):
        spin[mem_r[k], mem_c[k]] = s_new
    return csz


@njit(cache=True)
def _measure_ising(spin, L):
    N = L * L
    m = 0.0; e = 0.0
    for i in range(L):
        for j in range(L):
            m += spin[i, j]
            e -= spin[i, j] * (spin[(i + 1) % L, j] + spin[i, (j + 1) % L])
    return m / N, e / N


@njit(cache=True)
def _run_wolff(L, T, n_eq, n_meas, seed):
    np.random.seed(seed)
    p_add = 1.0 - np.exp(-2.0 / T)

    spin = np.ones((L, L))
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                spin[i, j] = -1.0

    for _ in range(n_eq):
        _wolff_step(spin, L, p_add)

    ms = 0.0; m2s = 0.0; m4s = 0.0; es = 0.0; e2s = 0.0
    for _ in range(n_meas):
        _wolff_step(spin, L, p_add)
        m, e = _measure_ising(spin, L)
        am = abs(m)
        ms += am; m2s += m*m; m4s += m**4; es += e; e2s += e*e

    return ms/n_meas, m2s/n_meas, m4s/n_meas, es/n_meas, e2s/n_meas


def _wolff_worker(args):
    L, T, n_eq, n_meas, seed = args
    return _run_wolff(L, T, n_eq, n_meas, seed)


def run_wolff_cpu(L, T_arr, n_eq=2000, n_meas=50000, n_workers=24, seed=42):
    # Warm up JIT compilation before spawning workers
    print("  Warming up numba JIT...", flush=True)
    _run_wolff(4, 2.269, 5, 5, 0)

    tasks = [(L, float(T), n_eq, n_meas, seed + i) for i, T in enumerate(T_arr)]
    with Pool(n_workers) as pool:
        results = list(pool.imap_unordered(_wolff_worker, tasks))
    # imap_unordered loses order; use imap instead
    with Pool(n_workers) as pool:
        results = list(pool.map(_wolff_worker, tasks))

    return dict(
        T  = T_arr,
        M  = np.array([r[0] for r in results]),
        M2 = np.array([r[1] for r in results]),
        M4 = np.array([r[2] for r in results]),
        E  = np.array([r[3] for r in results]),
        E2 = np.array([r[4] for r in results]),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    T_arr = np.linspace(1.6, 3.0, 100)
    data  = {}

    for L in [32, 64, 128, 256]:
        print(f"\n=== Metropolis GPU  L={L} ===")
        data[L] = run_metropolis_gpu(L, T_arr,
                                     n_eq=5000, n_meas=200000, batch_size=24)

    print(f"\n=== Wolff CPU  L=512 ===")
    data[512] = run_wolff_cpu(512, T_arr,
                               n_eq=2000, n_meas=50000, n_workers=24)

    np.save('ising_data.npy', data)
    print(f"\nSaved ising_data.npy   (Onsager T_c = {TC_ONSAGER:.6f})")


if __name__ == '__main__':
    main()
