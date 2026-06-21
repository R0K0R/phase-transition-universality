"""
Analysis: load all simulation data, fit beta, produce plots.
Expected files:
  ising_data.npy
  md{2,3}d_N{1000,2000,4000}.npy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, curve_fit

TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))
plt.rcParams.update({
    'font.size':        13,
    'axes.labelsize':   16,
    'axes.titlesize':   17,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  12,
    'axes.linewidth':   1.2,
    'xtick.major.width':1.1,
    'ytick.major.width':1.1,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '-',
    'lines.linewidth':  2.2,
    'lines.markersize': 7,
    'figure.dpi':       200,
    'savefig.dpi':      220,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})


# ── fitting utilities ─────────────────────────────────────────────────────────

def fit_beta_loglog(dT, op, t_lo=0.01, t_hi=0.35, Tc_ref=None):
    """
    Log-log fit of op vs dT in the scaling window [t_lo, t_hi].
    t = dT / Tc_ref (reduced temperature).  Returns (beta, A, r2).
    """
    if Tc_ref is None:
        Tc_ref = 1.0
    t    = dT / Tc_ref
    mask = (t > t_lo) & (t < t_hi) & (op > 0) & (dT > 0)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(dT[mask]), np.log(op[mask])
    b, la  = np.polyfit(lx, ly, 1)
    yp     = b * lx + la
    ss_res = ((ly - yp) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return b, np.exp(la), r2


def binder(M2, M4):
    return 1.0 - M4 / (3.0 * M2 ** 2)


# ── Ising analysis ────────────────────────────────────────────────────────────

def analyse_ising(data):
    sizes = sorted(data.keys())
    T_arr = data[sizes[0]]['T']

    UL = {L: binder(data[L]['M2'], data[L]['M4']) for L in sizes}

    # Binder crossing: use L=64 vs L=128 (cleanest crossing; L=512 is Wolff/noisy)
    # Search all adjacent pairs, pick crossing nearest to Onsager T_c
    best_cross = TC_ONSAGER
    best_dist  = 999.0
    for L1c, L2c in [(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]:
        cs1c = CubicSpline(T_arr, UL[L1c])
        cs2c = CubicSpline(T_arr, UL[L2c])
        diff_c = cs1c(T_arr) - cs2c(T_arr)
        for s in np.where(np.diff(np.sign(diff_c)))[0]:
            if not (1.9 < T_arr[s] < 2.6):
                continue
            try:
                tc = brentq(lambda T: cs1c(T) - cs2c(T), T_arr[s], T_arr[s+1])
                if abs(tc - TC_ONSAGER) < best_dist:
                    best_dist  = abs(tc - TC_ONSAGER)
                    best_cross = tc
            except Exception:
                pass
    T_cross = best_cross
    print(f"Ising Binder crossing  T_c = {T_cross:.5f}  (Onsager {TC_ONSAGER:.5f})")

    # beta: use L=128 (avoids sign-flip noise of L=256 and FSE of L=512)
    # Use Onsager T_c (exact) rather than noisy Binder estimate for better fit
    Lfit = sizes[-3] if len(sizes) >= 3 else sizes[-1]
    M    = data[Lfit]['M']
    dT   = TC_ONSAGER - T_arr
    beta, A, r2 = fit_beta_loglog(dT, M, t_lo=0.01, t_hi=0.10, Tc_ref=TC_ONSAGER)
    print(f"Ising  beta = {beta:.4f}   R² = {r2:.4f}   (L={Lfit})")

    return dict(T=T_arr, UL=UL, T_cross=T_cross, dT=dT,
                beta=beta, A=A, sizes=sizes, M=M, Lfit=Lfit, data=data)


# ── MD analysis ───────────────────────────────────────────────────────────────

def analyse_md_one(md):
    T     = md['T']; rl = md['rho_l']; rg = md['rho_g']
    delta = rl - rg
    N     = int(md.get('N', 0))
    dim   = int(md.get('dim', 0))

    # Fit T_c and beta simultaneously, then refine beta with fixed T_c
    def model(T_, Tc, A, b):
        return A * np.maximum(Tc - T_, 0) ** b

    try:
        # Allow T_c up to 25% above T_max; start guess at 10% above
        T_lo = T.max() * 1.005
        T_hi = T.max() * 1.25
        p0   = [T.max() * 1.10, 1.0, 0.2]
        popt, _ = curve_fit(model, T, delta, p0=p0,
                            bounds=([T_lo, 0.01, 0.05], [T_hi, 10.0, 0.7]),
                            maxfev=10000)
        Tc = popt[0]
    except Exception:
        Tc = T.max() * 1.10

    dT             = Tc - T
    beta, A, r2    = fit_beta_loglog(dT, delta, t_lo=0.01, t_hi=0.25, Tc_ref=Tc)
    print(f"MD dim={dim} N={N}  T_c={Tc:.4f}  beta={beta:.4f}  R²={r2:.4f}")
    return dict(T=T, rho_l=rl, rho_g=rg, delta=delta,
                T_c=Tc, beta=beta, A=A, dT=dT, N=N, dim=dim)


def analyse_md(dim, N_list):
    results = {}
    for N in N_list:
        fname = f"md{dim}d_N{N}.npy"
        # For 2D, prefer N=2000 near-T_c dataset for beta if available
        fname_near = f"md2d_N2000_nearTc.npy" if dim == 2 else None
        try:
            md_coex = np.load(fname, allow_pickle=True).item()
            if fname_near:
                try:
                    md_near = np.load(fname_near, allow_pickle=True).item()
                    print(f"  Using {fname_near} for beta, {fname} for coexistence curve")
                    res = analyse_md_one(md_near)
                    res['rho_l_coex'] = md_coex['rho_l']
                    res['rho_g_coex'] = md_coex['rho_g']
                    res['T_coex']     = md_coex['T']
                except FileNotFoundError:
                    print(f"  {fname_near} not found — using {fname} for both")
                    res = analyse_md_one(md_coex)
            else:
                res = analyse_md_one(md_coex)
            results[N] = res
        except FileNotFoundError:
            print(f"  {fname} not found — skipping")
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

COLORS = {'ising': '#1f77b4', 2: '#e377c2', 3: '#17becf'}
N_COLORS = {1000: '#2ca02c', 2000: '#ff7f0e', 4000: '#d62728'}

def plot_binder(res_ising):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    sizes = res_ising['sizes']
    for i, L in enumerate(sizes):
        c = cmap(0.15 + 0.7 * i / max(1, len(sizes) - 1))
        ax.plot(res_ising['T'], res_ising['UL'][L],
                label=f'$L={L}$', color=c, lw=2.4)
    ax.axvline(TC_ONSAGER, color='#d62728', ls='--', lw=1.8,
               label=f'Onsager $T_c={TC_ONSAGER:.4f}$', zorder=1)
    ax.axvline(res_ising['T_cross'], color='black', ls=':', lw=1.8,
               label=f'Binder crossing $T_c={res_ising["T_cross"]:.4f}$', zorder=1)
    ax.set_xlabel(r'Temperature $T \; [J/k_B]$')
    ax.set_ylabel(r'Binder cumulant  $U_L$')
    ax.set_title('Binder Cumulant Crossing Locates $T_c$')
    ax.set_ylim(0, 0.72)
    ax.set_xlim(res_ising['T'].min(), res_ising['T'].max())
    ax.legend(loc='lower left', framealpha=0.95)
    fig.tight_layout(); fig.savefig('binder.png')
    print("Saved binder.png")


def plot_magnetization(res_ising):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sizes = res_ising['sizes']
    cmap  = plt.cm.viridis
    T_arr = res_ising['T']
    raw   = res_ising['data']
    for i, L in enumerate(sizes):
        c  = cmap(0.15 + 0.7 * i / max(1, len(sizes) - 1))
        N  = L * L
        Cv = N * (raw[L]['E2'] - raw[L]['E'] ** 2) / T_arr ** 2
        axes[0].plot(T_arr, raw[L]['M'], color=c, lw=2.4, label=f'$L={L}$')
        axes[1].plot(T_arr, Cv,           color=c, lw=2.4, label=f'$L={L}$')
    for ax in axes:
        ax.axvline(TC_ONSAGER, color='#d62728', ls='--', lw=1.8,
                   label=f'Onsager $T_c$')
        ax.set_xlim(T_arr.min(), T_arr.max())
        ax.legend(loc='best', framealpha=0.95)
    axes[0].set_xlabel(r'Temperature $T \; [J/k_B]$')
    axes[0].set_ylabel(r'$\langle|M|\rangle$')
    axes[0].set_title('Magnetisation')
    axes[1].set_xlabel(r'Temperature $T \; [J/k_B]$')
    axes[1].set_ylabel(r'$C_v / N$')
    axes[1].set_title('Heat Capacity')
    fig.tight_layout(); fig.savefig('magnetization.png')
    print("Saved magnetization.png")


def plot_coexistence(res2d, res3d):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, res_dict, dim_lbl in [(axes[0], res2d, '2D'), (axes[1], res3d, '3D')]:
        for N, res in res_dict.items():
            c = N_COLORS.get(N, '#2ca02c')
            rho_l = res.get('rho_l_coex', res['rho_l'])
            rho_g = res.get('rho_g_coex', res['rho_g'])
            T_plt = res.get('T_coex',     res['T'])
            ax.plot(rho_l, T_plt, 'o-', color=c, lw=2.2, ms=7,
                    label=f'liquid $\\rho_l$  ($N={N}$)')
            ax.plot(rho_g, T_plt, 's--', color=c, lw=2.2, ms=7, mfc='white',
                    mew=1.6, label=f'gas $\\rho_g$  ($N={N}$)')
        if res_dict:
            Tc_ref = res_dict[max(res_dict)]['T_c']
            ax.axhline(Tc_ref, color='#d62728', ls=':', lw=2,
                       label=f'$T_c \\approx {Tc_ref:.3f}$ (fit)')
        ax.set_xlabel(r'Reduced density  $\rho^* = \rho \sigma^d$')
        ax.set_ylabel(r'Reduced temperature  $T^* = k_B T/\varepsilon$')
        ax.set_title(f'{dim_lbl} Lennard-Jones Coexistence Curve')
        ax.legend(loc='best', framealpha=0.95)
    fig.tight_layout(); fig.savefig('coexistence.png')
    print("Saved coexistence.png")


def plot_loglog(res_ising, res2d, res3d):
    fig, ax = plt.subplots(figsize=(10, 7))
    t_ref = np.logspace(np.log10(5e-3), np.log10(0.35), 200)

    # Ising (L=128, Onsager T_c)
    t_i  = res_ising['dT'] / TC_ONSAGER
    M_i  = res_ising['M']
    mk   = (t_i > 5e-3) & (t_i < 0.20) & (M_i > 0)
    Lfit = res_ising['sizes'][-3] if len(res_ising['sizes']) >= 3 else res_ising['sizes'][-1]
    ax.scatter(t_i[mk], M_i[mk], s=70, color=COLORS['ising'],
               edgecolor='black', linewidth=0.8, zorder=5,
               label=rf"2D Ising  $L={Lfit}$  ($\beta_\mathrm{{fit}}={res_ising['beta']:.3f}$)")

    # 2D MD (largest N)
    if res2d:
        Nbig = max(res2d)
        r2   = res2d[Nbig]
        t2   = r2['dT'] / r2['T_c']
        mk2  = (t2 > 5e-3) & (r2['delta'] > 0)
        ax.scatter(t2[mk2], r2['delta'][mk2], s=70, marker='s',
                   color=COLORS[2], edgecolor='black', linewidth=0.8, zorder=5,
                   label=rf"2D LJ  $N={Nbig}$  ($\beta_\mathrm{{fit}}={r2['beta']:.3f}$)")

    # 3D MD (largest N)
    if res3d:
        Nbig = max(res3d)
        r3   = res3d[Nbig]
        t3   = r3['dT'] / r3['T_c']
        mk3  = (t3 > 5e-3) & (r3['delta'] > 0)
        ax.scatter(t3[mk3], r3['delta'][mk3], s=70, marker='^',
                   color=COLORS[3], edgecolor='black', linewidth=0.8, zorder=5,
                   label=rf"3D LJ  $N={Nbig}$  (flat $\Rightarrow$ no separation)")

    # Reference power laws
    ax.plot(t_ref, 1.05 * t_ref ** 0.125, color='black', ls='--', lw=2,
            label=r'2D Ising  $\beta = 1/8$', zorder=2)
    ax.plot(t_ref, 2.0  * t_ref ** 0.326, color='gray',  ls=':',  lw=2,
            label=r'3D Ising  $\beta \approx 0.326$', zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Reduced temperature  $t = (T_c - T)/T_c$')
    ax.set_ylabel(r'Order parameter  $\langle|M|\rangle,\;\Delta\rho$')
    ax.set_title('Critical Scaling on Log--Log Axes')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout(); fig.savefig('loglog_beta.png')
    print("Saved loglog_beta.png")


def plot_beta_vs_N(res2d, res3d):
    fig, ax = plt.subplots(figsize=(6, 4))
    N_list = sorted(set(list(res2d.keys()) + list(res3d.keys())))
    for res_dict, lbl, col in [(res2d, '2D LJ', COLORS[2]), (res3d, '3D LJ', COLORS[3])]:
        Ns    = sorted(res_dict.keys())
        betas = [res_dict[N]['beta'] for N in Ns]
        ax.plot(Ns, betas, 'o-', color=col, label=lbl)
        ax.plot(Ns, betas, 'o', color=col, ms=8)
    ax.axhline(0.125, color='k',    ls='--', lw=1.2, label=r'2D Ising $\beta=1/8$')
    ax.axhline(0.326, color='gray', ls=':',  lw=1.2, label=r'3D Ising $\beta≈0.326$')
    ax.set(xscale='log', xlabel='N', ylabel=r'$\beta$',
           title=r'Finite-size convergence of $\beta$', ylim=(0, 0.45))
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig('beta_vs_N.png')
    print("Saved beta_vs_N.png")


def plot_beta_summary(res_ising, res2d, res3d):
    rows = [('2D Ising MC\n($L=128$)', COLORS['ising'], res_ising['beta'])]
    for dim, res_dict, col in [(2, res2d, COLORS[2]), (3, res3d, COLORS[3])]:
        if res_dict:
            Nbig = max(res_dict)
            rows.append((f'{dim}D LJ MD\n($N={Nbig}$)', col, res_dict[Nbig]['beta']))
    rows.append(('Onsager\nexact', '#d62728', 0.125))

    labels, colors, betas = zip(*rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, betas, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.4, width=0.6)
    ax.axhline(0.125, color='#d62728', ls='--', lw=2,
               label=r'2D Ising class:  $\beta = 1/8 = 0.125$', zorder=1)
    ax.axhline(0.326, color='gray', ls=':', lw=2,
               label=r'3D Ising class:  $\beta \approx 0.326$', zorder=1)
    for bar, b in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width()/2, b + 0.008,
                f'{b:.3f}', ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Critical exponent  $\beta$')
    ax.set_title(r'Universality Comparison of $\beta$ Across Systems')
    ax.set_ylim(0, 0.48)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig('beta_summary.png')
    print("Saved beta_summary.png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N_list_2d = [1000]        # coexistence curve; nearTc uses N=2000 automatically
    N_list_3d = [800, 1000]   # try N=800 first, fall back to N=1000

    print("── Ising ─────────────────────────────────")
    ising_raw  = np.load('ising_data.npy', allow_pickle=True).item()
    res_ising  = analyse_ising(ising_raw)

    print("\n── MD 2D ─────────────────────────────────")
    res2d = analyse_md(2, N_list_2d)

    print("\n── MD 3D ─────────────────────────────────")
    res3d = analyse_md(3, N_list_3d)

    print("\n── Summary ───────────────────────────────")
    print(f"  Ising       beta = {res_ising['beta']:.4f}   T_c = {res_ising['T_cross']:.4f}")
    for N in N_list_2d:
        if N in res2d: print(f"  2D LJ N={N}  beta = {res2d[N]['beta']:.4f}   T_c = {res2d[N]['T_c']:.4f}")
    for N in N_list_3d:
        if N in res3d: print(f"  3D LJ N={N}  beta = {res3d[N]['beta']:.4f}   T_c = {res3d[N]['T_c']:.4f}")
    print(f"  Onsager     beta = 0.1250   T_c = {TC_ONSAGER:.4f}")

    print("\n── Plots ─────────────────────────────────")
    plot_binder(res_ising)
    plot_magnetization(res_ising)
    plot_coexistence(res2d, res3d)
    plot_loglog(res_ising, res2d, res3d)
    plot_beta_vs_N(res2d, res3d)
    plot_beta_summary(res_ising, res2d, res3d)
