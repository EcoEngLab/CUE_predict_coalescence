import os

# Parallelize across seeds rather than also spawning BLAS threads in each worker.
# Set defaults before importing NumPy/SciPy; explicit environment overrides remain valid.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
):
    os.environ.setdefault(_thread_variable, "1")

from multiprocessing import Pool, TimeoutError as PoolTimeoutError, cpu_count
from time import perf_counter
import numpy as np
import pandas as pd
import sys
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


# Basic simulation of microbial community coalescence


# =========================
# 1. Parameter settings
# =========================
code_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_path)

# Random seed settings
BASE_SEED = 37
N_SIMULATIONS = 50
N_WORKERS = min(4, cpu_count())

# Extra monoculture assays match rmax_cue.py: Community 1 species only.
# Adds N1 small ODE solves per seed; disable to skip monoculture assays.
COMPUTE_MEASURABLE_CUE = True
EIGENVALUE_TOL = 1e-8

# Species pool and resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1.0
LEAKAGE_RATE = 0.2

# Community parameters
N1, M1 = 100, 50
N2, M2 = 100, 50

# Dynamical parameters
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
T_SPAN = (0, 100000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Mechanistic curve parameter settings
THEORY_LOCAL_Q = 0.35
MIN_POINTS_FOR_THEORY = 5


# =========================
# 2. Core functions
# =========================
def modular_uptake(N, M, N_modules, s_ratio, rng):
    assert N_modules <= M and N_modules <= N, "N_modules must be no larger than both M and N"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    diffC = np.full(N_modules, sC, dtype=int)
    if dC > 0:
        diffC[rng.choice(N_modules, dC, replace=False)] += 1
    mC = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))]

    u = rng.random((N, M))

    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(u, axis=1, keepdims=True)
    u = u / row_sums
    return u


def modular_leakage(M, N_modules, s_ratio, lam, rng):
    assert N_modules <= M, "N_modules must be no larger than M"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [list(range(x - 1, y)) for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))]

    l = rng.random((M, M))

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(l, axis=1, keepdims=True)
    l = lam * l / row_sums
    return l


def generate_l_tensor(N, M, N_modules, s_ratio, lam, rng):
    l_tensor = np.empty((N, M, M))
    for i in range(N):
        l_tensor[i] = modular_leakage(M, N_modules, s_ratio, lam, rng)
    return l_tensor


def safe_weighted_average(values, weights):
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan
    return np.sum(values * weights) / total_weight


def compute_eta_from_l(l):
    return 1.0 - np.sum(l, axis=2)


def ensure_m_vector(m, N):
    if np.ndim(m) == 0:
        return np.full(N, float(m))
    m_vec = np.asarray(m, dtype=float)
    if m_vec.shape[0] != N:
        raise ValueError("Length of m does not match N")
    return m_vec


def compute_Gi0_Ui0_eps(u, l, R0, m):
    N = u.shape[0]
    eta = compute_eta_from_l(l)
    m_vec = ensure_m_vector(m, N)
    Ui0 = np.sum(u * R0[None, :], axis=1)
    Gi0 = np.sum(u * eta * R0[None, :], axis=1)
    eps = (Gi0 - m_vec) / (Ui0 + 1e-12)
    return eta, Gi0, Ui0, eps


def compute_CUE(sol, N, u, R_ref, l, m):
    C_values = sol.y[:N, -1]
    _, Gi0, Ui0, species_CUE = compute_Gi0_Ui0_eps(u, l, R_ref, m)
    community_CUE = safe_weighted_average(species_CUE, C_values)
    return community_CUE, species_CUE


def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method='BDF', dense_output=False
):
    m_vec = ensure_m_vector(m, N)
    rho = np.asarray(rho, dtype=float)
    omega = np.asarray(omega, dtype=float)
    C0 = np.asarray(C0, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    eta = compute_eta_from_l(l)

    def dCdt_Rdt(t, y):
        C = np.clip(y[:N], 0.0, None)
        R = np.clip(y[N:], 0.0, None)

        growth_flux = u * eta * R[None, :]
        dCdt = C * (np.sum(growth_flux, axis=1) - m_vec)

        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R[None, :], axis=0)
        dRdt -= consumption

        leakage = np.einsum('i,b,ib,iba->a', C, R, u, l, optimize='optimal')
        dRdt += leakage

        return np.concatenate([dCdt, dRdt])

    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y)
        return np.max(np.abs(deriv)) - tol

    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    Y0 = np.concatenate([C0, R0])

    solve_kwargs = dict(
        fun=dCdt_Rdt,
        t_span=t_span,
        y0=Y0,
        method=method,
        events=equilibrium_event,
        dense_output=dense_output
    )
    if t_eval is not None:
        solve_kwargs["t_eval"] = t_eval

    sol = solve_ivp(**solve_kwargs)
    return sol


def micrm_jacobian(C, R, u, l, m, omega):
    """Analytic Jacobian of the eta-based model, including extinct species.

    l[i, b, a] routes resource b into resource a. At a zero population this
    uses the ecological invasion derivative, rather than differentiating clip().
    """
    retained = u * compute_eta_from_l(l)
    growth = retained @ R - ensure_m_vector(m, len(C))
    Jcc = np.diag(growth)
    Jcr = C[:, None] * retained
    Jrc = (np.einsum('ib,iba->ia', u * R, l) - u * R).T
    Jrr = np.einsum('i,ib,iba->ab', C, u, l) - np.diag(omega + C @ u)
    return np.block([[Jcc, Jcr], [Jrc, Jrr]])


def effective_lv_parameters(C, R, u, l, m, omega, jacobian=None):
    """Linearize resource equilibrium: g(C) ~= r + alpha @ C."""
    N = len(C)
    J = micrm_jacobian(C, R, u, l, m, omega) if jacobian is None else jacobian
    resource_response = np.linalg.solve(-J[N:, N:], J[N:, :N])
    retained = u * compute_eta_from_l(l)
    alpha = retained @ resource_response
    r = retained @ R - ensure_m_vector(m, N) - alpha @ C
    return alpha, r


def feasibility_proxy(alpha, cond_max=1e10):
    """SVD log-volume proxy per survivor dimension; NOT a probability.

    Matches the donor's log10 scale, with adaptive positive diagonal ridge.
    Negative values are valid. Report ridge size and conditioning explicitly.
    """
    result = {"feasibility": np.nan, "Feasibility_Ridge": np.nan,
              "Feasibility_Condition": np.nan, "Feasibility_Status": "no_survivors"}
    if alpha.shape[0] == 0:
        return result
    if not np.all(np.isfinite(alpha)) or not np.any(alpha):
        result["Feasibility_Status"] = "invalid_matrix"
        return result
    diagonal = np.abs(np.diag(alpha))
    positive = diagonal[diagonal > 0]
    scale = max(float(np.median(positive if positive.size else np.abs(alpha[alpha != 0]))), 1e-12)
    identity = np.eye(len(alpha))

    def candidate(relative):
        A = alpha + relative * scale * identity
        singular = np.linalg.svd(A, compute_uv=False)
        condition = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
        return singular, condition

    try:
        relative, low = 1e-10, 1e-11
        singular, condition = candidate(relative)
        while (not np.isfinite(condition) or condition > cond_max) and relative < 1e-2:
            low, relative = relative, min(relative * 10, 1e-2)
            singular, condition = candidate(relative)
        if not np.isfinite(condition) or condition > cond_max:
            result["Feasibility_Status"] = "ill_conditioned"
            return result
        for _ in range(20):
            middle = np.sqrt(low * relative)
            trial_singular, trial_condition = candidate(middle)
            if np.isfinite(trial_condition) and trial_condition <= cond_max:
                relative, singular, condition = middle, trial_singular, trial_condition
            else:
                low = middle
        result.update(feasibility=float(-np.mean(np.log10(singular))),
                      Feasibility_Ridge=float(relative * scale),
                      Feasibility_Condition=float(condition), Feasibility_Status="ok")
    except np.linalg.LinAlgError:
        result["Feasibility_Status"] = "svd_failed"
    return result


def micrm_depletion_competition_matrix(C_hat, R_hat, u, l_tensor, omega):
    """Direct resource-depletion coefficients from equations (7)-(8).

    q[i,a] = u[i,a] * (1 - sum_b l[i,a,b])
    A_dep[i,j] = sum_a q[i,a] * u[j,a] * R*[a]
                         / (omega[a] + sum_s u[s,a] * C*[s])

    All populations contribute to the resource-loss denominator. This is the
    diagonal resource-response expression in the supplied equations; leakage
    enters q, without an additional inverse leakage-feedback matrix.
    """
    abundance = np.asarray(C_hat, dtype=float)
    resources = np.asarray(R_hat, dtype=float)
    uptake = np.asarray(u, dtype=float)
    leakage = np.asarray(l_tensor, dtype=float)
    n_species, n_resources = uptake.shape
    if abundance.shape != (n_species,) or resources.shape != (n_resources,):
        raise ValueError("C_hat and R_hat must match the dimensions of u")
    if leakage.shape != (n_species, n_resources, n_resources):
        raise ValueError("l_tensor must have shape (species, resources, resources)")
    turnover = np.broadcast_to(np.asarray(omega, dtype=float), (n_resources,))
    resource_loss = turnover + abundance @ uptake
    if not np.all(np.isfinite(resource_loss)) or np.any(resource_loss <= 0):
        raise ValueError("Resource-loss denominators must be finite and positive")
    retained_uptake = uptake * compute_eta_from_l(leakage)
    matrix = (retained_uptake * (resources / resource_loss)[None, :]) @ uptake.T
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Depletion competition matrix must be finite")
    return matrix


def community_heterospecific_competition_pressure(matrix, C_final, survivor_idx=None):
    """Equation (9): abundance-weighted pressure from other surviving species.

    Restrict both focal species i and competitors j to survivors, exclude i=j,
    and divide by total survivor abundance. One survivor gives zero; none NaN.
    """
    matrix = np.asarray(matrix, dtype=float)
    abundance = np.asarray(C_final, dtype=float)
    if matrix.shape != (abundance.size, abundance.size):
        raise ValueError("Competition matrix must be square and match C_final")
    if survivor_idx is None:
        idx = np.flatnonzero(abundance > SURVIVAL_THRESHOLD)
    else:
        idx = np.asarray(survivor_idx)
        if idx.dtype == bool:
            if idx.shape != abundance.shape:
                raise ValueError("Survivor mask must match C_final")
            idx = np.flatnonzero(idx)
        else:
            idx = idx.astype(int)
    if idx.size == 0:
        return np.nan
    submatrix = matrix[np.ix_(idx, idx)].copy()
    np.fill_diagonal(submatrix, 0.0)
    survivor_abundance = abundance[idx]
    total_abundance = np.sum(survivor_abundance)
    if total_abundance <= 0:
        return np.nan
    pressure = submatrix @ survivor_abundance
    return float(survivor_abundance @ pressure / total_abundance)


def analyze_community(sol, u, l, m, rho, omega):
    N = len(u)
    C, R = np.maximum(sol.y[:N, -1], 0), np.maximum(sol.y[N:, -1], 0)
    survivors = C > SURVIVAL_THRESHOLD
    eta = compute_eta_from_l(l)
    J = micrm_jacobian(C, R, u, l, m, omega)
    growth = (u * eta) @ R - ensure_m_vector(m, N)
    resource_derivative = rho - omega * R + J[N:, :N] @ C
    residual = float(np.max(np.abs(np.concatenate([C * growth, resource_derivative]))))
    at_equilibrium = bool(sol.success and residual <= 1.01e-5)
    leading = float(np.max(np.linalg.eigvals(J).real)) if sol.success else np.nan
    if not sol.success:
        status = "solver_failed"
    elif not at_equilibrium:
        status = "not_equilibrated"
    elif leading < -EIGENVALUE_TOL:
        status = "stable"
    elif leading > EIGENVALUE_TOL:
        status = "unstable"
    else:
        status = "marginal"
    try:
        alpha, intrinsic_growth = effective_lv_parameters(C, R, u, l, m, omega, J)
        feasibility = feasibility_proxy(alpha[np.ix_(survivors, survivors)])
    except np.linalg.LinAlgError:
        intrinsic_growth = np.full(N, np.nan)
        feasibility = {"feasibility": np.nan, "Feasibility_Ridge": np.nan,
                       "Feasibility_Condition": np.nan, "Feasibility_Status": "resource_matrix_singular"}
    # Evaluate the resource-mediated competition only at a numerical equilibrium.
    depletion_pressure = np.nan
    if not sol.success:
        depletion_status = "solver_failed"
    elif not at_equilibrium:
        depletion_status = "not_equilibrated"
    elif not np.any(survivors):
        depletion_status = "no_survivors"
    else:
        try:
            depletion_matrix = micrm_depletion_competition_matrix(C, R, u, l, omega)
            depletion_pressure = community_heterospecific_competition_pressure(
                depletion_matrix, C, survivor_idx=survivors)
            depletion_status = "ok"
        except ValueError:
            depletion_status = "invalid_resource_response"

    metrics = dict(Heterospecific_Competition_Pressure=depletion_pressure,
                   Depletion_Competition_Status=depletion_status,
                   Leading_Eigenvalue=leading, N_Survivors=int(np.sum(survivors)),
                   Equilibrium_Residual=residual, Equilibrium_Reached=at_equilibrium,
                   Integration_Success=bool(sol.success), End_Time=float(sol.t[-1]),
                   Stability_Status=status, **feasibility)
    return intrinsic_growth, metrics


def measurable_cue_monoculture(u_i, l_i, m, rho, omega, C0, R0, t_span):
    """Donor rmax/(rmax+m) proxy from one species grown alone.

    rmax is the largest per-capita net growth rate, not max(dC/dt).
    This proxy is distinct from net production / gross resource uptake.
    """
    M = u_i.shape[1]
    sol = solve_micrm(1, M, u_i, l_i, m, np.full(M, LEAKAGE_RATE),
                      rho, omega, np.array([C0]), R0, t_span, dense_output=True)
    result = dict(rmax=np.nan, t_star=np.nan, growth_CUE=np.nan,
                  Monoculture_End_Time=float(sol.t[-1]), Monoculture_Success=bool(sol.success))
    if not sol.success or len(sol.t) < 2:
        return result
    retained = (u_i * compute_eta_from_l(l_i))[0]
    mu = retained @ np.maximum(sol.y[1:], 0) - m
    peak = int(np.argmax(mu))
    time_star, rmax = float(sol.t[peak]), float(mu[peak])
    # Refine the sampled peak using the solver interpolant, including endpoints.
    left, right = sol.t[max(0, peak - 1)], sol.t[min(len(sol.t) - 1, peak + 1)]
    if right > left:
        fit = minimize_scalar(lambda t: -(retained @ np.maximum(sol.sol(t)[1:], 0) - m),
                              bounds=(left, right), method="bounded")
        if fit.success and -fit.fun > rmax:
            time_star, rmax = float(fit.x), float(-fit.fun)
    result.update(rmax=rmax, t_star=time_star,
                  growth_CUE=rmax / (rmax + m) if rmax > 0 and rmax + m > 0 else np.nan)
    return result


def calculate_effective_leakage(u, l):
    return np.einsum('ia,iab->ib', u, l, optimize='optimal')


def community_level_competition(u):
    N, _ = u.shape
    if N < 2:
        return np.nan

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T

    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += similarity[i, j]

    return 2 * total / (N * (N - 1))


def species_level_competition(u):
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_normalized = u / (norms + 1e-10)
    similarity = u_normalized @ u_normalized.T
    np.fill_diagonal(similarity, 0.0)

    comp = np.sum(similarity, axis=1) / (N - 1)
    return comp


def species_level_competition_dot(u):
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    comp_matrix = u @ u.T
    np.fill_diagonal(comp_matrix, 0.0)
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp


def compute_uptake_variance(u):
    return np.var(u, axis=1)


def choose_resources_for_second_community(M_pool, M1, M2, resource_indices1, rng):
    if M1 > M2:
        return rng.choice(resource_indices1, M2, replace=False)
    elif M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = rng.choice(remaining_resources, M2 - M1, replace=False)
        return np.concatenate([resource_indices1, additional_resources])
    else:
        return resource_indices1.copy()


# =========================
# 3. Theoretical curve functions
# =========================
def cue_abundance_theory(eps, eps_c, H, Cmax):
    eps = np.asarray(eps, dtype=float)
    if not np.isfinite(eps_c) or not np.isfinite(H) or not np.isfinite(Cmax):
        return np.full_like(eps, np.nan, dtype=float)
    H = max(float(H), 1e-12)
    Cmax = max(float(Cmax), 1e-12)
    delta = np.maximum(eps - eps_c, 0.0)
    return Cmax * (1.0 - np.exp(-delta / H))


def gi_of_R(u_i, eta_i, R):
    return np.sum(u_i * eta_i * R)


def solve_resident_environment_for_species(
    species_idx, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span
):
    keep = np.arange(N) != species_idx
    N_res = int(np.sum(keep))

    if N_res == 0:
        rho = np.asarray(rho, dtype=float)
        omega = np.asarray(omega, dtype=float)
        return rho / np.maximum(omega, 1e-12)

    u_res = u[keep]
    l_res = l[keep]
    m_res = ensure_m_vector(m, N)[keep]
    C0_res = np.maximum(C_full[keep], 1e-12)
    R0_res = np.maximum(R_full, 1e-12)

    sol_res = solve_micrm(
        N=N_res,
        M=M,
        u=u_res,
        l=l_res,
        m=m_res,
        lambda_alpha=lambda_alpha,
        rho=rho,
        omega=omega,
        C0=C0_res,
        R0=R0_res,
        t_span=t_span,
        t_eval=None
    )
    R_res = np.maximum(sol_res.y[N_res:, -1], 0.0)
    return R_res


def compute_mechanistic_curve_params_for_one_realization(
    N, M, u, l, m, lambda_alpha, rho, omega, R0_ref, C_full, R_full, t_span,
    survival_threshold=1e-5, local_q=0.35
):
    eta, Gi0, Ui0, eps = compute_Gi0_Ui0_eps(u, l, R0_ref, m)

    eps_c = np.full(N, np.nan)
    gi_res = np.full(N, np.nan)
    gi_full = np.full(N, np.nan)
    D_obs = np.full(N, np.nan)
    chi_obs = np.full(N, np.nan)

    for i in range(N):
        R_res_i = solve_resident_environment_for_species(
            i, N, M, u, l, m, lambda_alpha, rho, omega, C_full, R_full, t_span
        )

        gi_res[i] = gi_of_R(u[i], eta[i], R_res_i)
        gi_full[i] = gi_of_R(u[i], eta[i], R_full)

        eps_c[i] = (Gi0[i] - gi_res[i]) / (Ui0[i] + 1e-12)
        D_obs[i] = gi_res[i] - gi_full[i]

        if (C_full[i] > survival_threshold) and np.isfinite(D_obs[i]) and (D_obs[i] > 0):
            chi_obs[i] = D_obs[i] / max(C_full[i], 1e-12)

    delta_eps = np.maximum(eps - eps_c, 0.0)

    valid = (
        np.isfinite(chi_obs) &
        np.isfinite(delta_eps) &
        np.isfinite(C_full) &
        (delta_eps > 0) &
        (C_full > survival_threshold) &
        (D_obs > 0)
    )

    near_mask = valid.copy()
    if np.sum(valid) >= MIN_POINTS_FOR_THEORY:
        delta_cut = np.quantile(delta_eps[valid], local_q)
        abund_cut = np.quantile(C_full[valid], local_q)
        near_mask = valid & (delta_eps <= delta_cut) & (C_full <= abund_cut)
        if np.sum(near_mask) < 3:
            near_mask = valid

    chi_bar = np.nanmedian(chi_obs[near_mask]) if np.any(near_mask) else np.nan
    U_bar = np.nanmedian(Ui0[near_mask]) if np.any(near_mask) else np.nanmedian(Ui0[np.isfinite(Ui0)])
    eps_c_bar = np.nanmedian(eps_c[np.isfinite(eps_c)])
    Cmax = np.nanmax(C_full[np.isfinite(C_full)]) if np.any(np.isfinite(C_full)) else np.nan

    H = np.nan
    if np.isfinite(chi_bar) and np.isfinite(U_bar) and np.isfinite(Cmax) and (chi_bar > 0) and (U_bar > 0) and (Cmax > 0):
        H = chi_bar * Cmax / U_bar

    y_pred = cue_abundance_theory(eps, eps_c_bar, H, Cmax)
    surv_mask = np.isfinite(C_full) & (C_full > survival_threshold)
    if np.any(surv_mask) and np.all(np.isfinite(y_pred[surv_mask])):
        log_obs = np.log10(np.maximum(C_full[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res = np.sum((log_obs - log_pred) ** 2)
        ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
        theory_R2_log = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    else:
        theory_R2_log = np.nan

    species_df = pd.DataFrame({
        "Gi0": Gi0,
        "Ui0": Ui0,
        "eps_c_i": eps_c,
        "Delta_eps_i": delta_eps,
        "gi_res_i": gi_res,
        "gi_full_i": gi_full,
        "D_obs_i": D_obs,
        "chi_i_obs": chi_obs
    })

    params = {
        "eps_c": eps_c_bar,
        "chi_bar": chi_bar,
        "U_bar": U_bar,
        "Cmax": Cmax,
        "H": H,
        "Theory_R2_log": theory_R2_log,
        "NearThresholdUsed": int(np.sum(near_mask)),
        "N_survivors": int(np.sum(surv_mask))
    }
    return species_df, params


def estimate_theory_params_mechanistic(df_comm, survival_threshold=1e-5):
    dat = df_comm.copy()
    dat = dat[np.isfinite(dat["Species_CUE"]) & np.isfinite(dat["Abundance"])]

    if len(dat) < MIN_POINTS_FOR_THEORY:
        return None

    required_cols = [
        "Theory_eps_c_seed", "Theory_chi_bar_seed", "Theory_U_bar_seed",
        "Theory_Cmax_seed", "Theory_H_seed"
    ]
    if not all(col in dat.columns for col in required_cols):
        return None

    seed_params = (
        dat.groupby("Seed", as_index=False)
        .agg(
            eps_c=("Theory_eps_c_seed", "first"),
            chi_bar=("Theory_chi_bar_seed", "first"),
            U_bar=("Theory_U_bar_seed", "first"),
            Cmax=("Theory_Cmax_seed", "first"),
            H_seed=("Theory_H_seed", "first"),
            Theory_R2_log_seed=("Theory_R2_log_seed", "first"),
            NearThresholdUsed=("Theory_NearThresholdUsed_seed", "first")
        )
    )

    eps_c = np.nanmedian(seed_params["eps_c"])
    chi_bar = np.nanmedian(seed_params["chi_bar"])
    U_bar = np.nanmedian(seed_params["U_bar"])
    Cmax = np.nanmedian(seed_params["Cmax"])

    if not np.isfinite(eps_c) or not np.isfinite(chi_bar) or not np.isfinite(U_bar) or not np.isfinite(Cmax):
        return None
    if chi_bar <= 0 or U_bar <= 0 or Cmax <= 0:
        return None

    H = chi_bar * Cmax / U_bar

    x = dat["Species_CUE"].to_numpy()
    y = dat["Abundance"].to_numpy()
    y_pred = cue_abundance_theory(x, eps_c, H, Cmax)

    surv_mask = y > survival_threshold
    if np.any(surv_mask):
        log_obs = np.log10(np.maximum(y[surv_mask], survival_threshold))
        log_pred = np.log10(np.maximum(y_pred[surv_mask], survival_threshold))
        ss_res = np.sum((log_obs - log_pred) ** 2)
        ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
        theory_R2_log = np.nan if ss_tot <= 0 else 1 - ss_res / ss_tot
    else:
        theory_R2_log = np.nan

    return {
        "eps_c": eps_c,
        "chi_bar": chi_bar,
        "U_bar": U_bar,
        "H": H,
        "Cmax": Cmax,
        "Theory_R2_log": theory_R2_log,
        "N_total": len(dat),
        "N_survivors": int(np.sum(surv_mask)),
        "N_seeds": len(seed_params),
        "NearThresholdUsed_median": np.nanmedian(seed_params["NearThresholdUsed"])
    }


# =========================
# 4. Once-per-seed simulation function
# =========================
def simulate(seed):
    rng = np.random.default_rng(seed)

    u_pool = modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, rng)

    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)

    resource_indices2 = choose_resources_for_second_community(M_POOL, M1, M2, resource_indices1, rng)
    remaining_species = np.setdiff1d(np.arange(N_POOL), species_indices1)
    species_indices2 = rng.choice(remaining_species, N2, replace=False)

    u2 = u_pool[np.ix_(species_indices2, resource_indices2)]
    l2 = l_pool[np.ix_(species_indices2, resource_indices2, resource_indices2)]

    lambda_alpha2 = np.full(M2, LEAKAGE_RATE)
    rho2 = np.full(M2, RHO_VALUE)
    omega2 = np.full(M2, OMEGA_VALUE)
    C0_2 = np.full(N2, C0_VALUE)
    R0_2 = np.full(M2, R0_VALUE)

    sol1 = solve_micrm(N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1, C0_1, R0_1, T_SPAN)
    sol2 = solve_micrm(N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2, C0_2, R0_2, T_SPAN)

    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

    N3 = N1 + N2
    M3 = len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M3, R0_VALUE)

    sol3 = solve_micrm(N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3, C0_3, R0_3, T_SPAN)

    community_CUE1, species_CUE1 = compute_CUE(sol1, N1, u1, R0_1, l1, MAINTENANCE_COST)
    community_CUE2, species_CUE2 = compute_CUE(sol2, N2, u2, R0_2, l2, MAINTENANCE_COST)
    community_CUE3, species_CUE3 = compute_CUE(sol3, N3, u3, R0_3, l3, MAINTENANCE_COST)

    C_final1 = np.maximum(sol1.y[:N1, -1], 0.0)
    C_final2 = np.maximum(sol2.y[:N2, -1], 0.0)
    C_final3 = np.maximum(sol3.y[:N3, -1], 0.0)

    R_final1 = np.maximum(sol1.y[N1:, -1], 0.0)
    R_final2 = np.maximum(sol2.y[N2:, -1], 0.0)
    R_final3 = np.maximum(sol3.y[N3:, -1], 0.0)

    mech1, params1 = compute_mechanistic_curve_params_for_one_realization(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1, rho1, omega1,
        R0_1, C_final1, R_final1, T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )
    mech2, params2 = compute_mechanistic_curve_params_for_one_realization(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2, rho2, omega2,
        R0_2, C_final2, R_final2, T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )
    mech3, params3 = compute_mechanistic_curve_params_for_one_realization(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3, rho3, omega3,
        R0_3, C_final3, R_final3, T_SPAN, SURVIVAL_THRESHOLD, THEORY_LOCAL_Q
    )

    pred1 = cue_abundance_theory(species_CUE1, params1["eps_c"], params1["H"], params1["Cmax"])
    pred2 = cue_abundance_theory(species_CUE2, params2["eps_c"], params2["H"], params2["Cmax"])
    pred3 = cue_abundance_theory(species_CUE3, params3["eps_c"], params3["H"], params3["Cmax"])

    survivors1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    community_CUE1_surv = safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2_surv = safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3_surv = safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])

    L_eff1 = calculate_effective_leakage(u1, l1)
    L_eff2 = calculate_effective_leakage(u2, l2)
    L_eff3 = calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(L_eff1, axis=1)
    facilitation2 = np.mean(L_eff2, axis=1)
    facilitation3 = np.mean(L_eff3, axis=1)

    competition_comm1 = community_level_competition(u1)
    competition_comm2 = community_level_competition(u2)
    competition_comm3 = community_level_competition(u3)

    competition_species1 = species_level_competition(u1)
    competition_species2 = species_level_competition(u2)
    competition_species3 = species_level_competition(u3)

    competition_dot1 = species_level_competition_dot(u1)
    competition_dot2 = species_level_competition_dot(u2)
    competition_dot3 = species_level_competition_dot(u3)

    uptake_var1 = compute_uptake_variance(u1)
    uptake_var2 = compute_uptake_variance(u2)
    uptake_var3 = compute_uptake_variance(u3)

    depletion1 = np.sum(R_final1)
    depletion2 = np.sum(R_final2)
    depletion3 = np.sum(R_final3)

    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community 1" if origin1_in_coalesced > origin2_in_coalesced else "Community 2"

    species_data = []

    for i in range(N1):
        species_data.append({
            "Seed": seed,
            "Community": 1,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Community_CUE_surv": community_CUE1_surv,
            "Abundance": C_final1[i],
            "Theory_Abundance": pred1[i],
            "Theory_DeltaEps": max(species_CUE1[i] - params1["eps_c"], 0.0),
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i],
            "Gi0": mech1.loc[i, "Gi0"],
            "Ui0": mech1.loc[i, "Ui0"],
            "eps_c_i": mech1.loc[i, "eps_c_i"],
            "Delta_eps_i": mech1.loc[i, "Delta_eps_i"],
            "gi_res_i": mech1.loc[i, "gi_res_i"],
            "gi_full_i": mech1.loc[i, "gi_full_i"],
            "D_obs_i": mech1.loc[i, "D_obs_i"],
            "chi_i_obs": mech1.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": params1["eps_c"],
            "Theory_chi_bar_seed": params1["chi_bar"],
            "Theory_U_bar_seed": params1["U_bar"],
            "Theory_Cmax_seed": params1["Cmax"],
            "Theory_H_seed": params1["H"],
            "Theory_R2_log_seed": params1["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": params1["NearThresholdUsed"]
        })

    for i in range(N2):
        species_data.append({
            "Seed": seed,
            "Community": 2,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Community_CUE_surv": community_CUE2_surv,
            "Abundance": C_final2[i],
            "Theory_Abundance": pred2[i],
            "Theory_DeltaEps": max(species_CUE2[i] - params2["eps_c"], 0.0),
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i],
            "Gi0": mech2.loc[i, "Gi0"],
            "Ui0": mech2.loc[i, "Ui0"],
            "eps_c_i": mech2.loc[i, "eps_c_i"],
            "Delta_eps_i": mech2.loc[i, "Delta_eps_i"],
            "gi_res_i": mech2.loc[i, "gi_res_i"],
            "gi_full_i": mech2.loc[i, "gi_full_i"],
            "D_obs_i": mech2.loc[i, "D_obs_i"],
            "chi_i_obs": mech2.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": params2["eps_c"],
            "Theory_chi_bar_seed": params2["chi_bar"],
            "Theory_U_bar_seed": params2["U_bar"],
            "Theory_Cmax_seed": params2["Cmax"],
            "Theory_H_seed": params2["H"],
            "Theory_R2_log_seed": params2["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": params2["NearThresholdUsed"]
        })

    for i in range(N3):
        species_data.append({
            "Seed": seed,
            "Community": 3,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Community_CUE_surv": community_CUE3_surv,
            "Abundance": C_final3[i],
            "Theory_Abundance": pred3[i],
            "Theory_DeltaEps": max(species_CUE3[i] - params3["eps_c"], 0.0),
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i],
            "Gi0": mech3.loc[i, "Gi0"],
            "Ui0": mech3.loc[i, "Ui0"],
            "eps_c_i": mech3.loc[i, "eps_c_i"],
            "Delta_eps_i": mech3.loc[i, "Delta_eps_i"],
            "gi_res_i": mech3.loc[i, "gi_res_i"],
            "gi_full_i": mech3.loc[i, "gi_full_i"],
            "D_obs_i": mech3.loc[i, "D_obs_i"],
            "chi_i_obs": mech3.loc[i, "chi_i_obs"],
            "Theory_eps_c_seed": params3["eps_c"],
            "Theory_chi_bar_seed": params3["chi_bar"],
            "Theory_U_bar_seed": params3["U_bar"],
            "Theory_Cmax_seed": params3["Cmax"],
            "Theory_H_seed": params3["H"],
            "Theory_R2_log_seed": params3["Theory_R2_log"],
            "Theory_NearThresholdUsed_seed": params3["NearThresholdUsed"]
        })

    # Add analyses without changing existing abundance-weighted CUE columns.
    analyses = {}
    for community, sol, u, l, rho, omega in (
        (1, sol1, u1, l1, rho1, omega1),
        (2, sol2, u2, l2, rho2, omega2),
        (3, sol3, u3, l3, rho3, omega3),
    ):
        analyses[community] = analyze_community(sol, u, l, MAINTENANCE_COST, rho, omega)
    for row in species_data:
        intrinsic_growth, metrics = analyses[row["Community"]]
        index = row["Species_ID"] - 1
        row.update(metrics)
        row["Growth_Rate"] = intrinsic_growth[index]
        row.update(rmax=np.nan, t_star=np.nan, growth_CUE=np.nan,
                   Monoculture_End_Time=np.nan, Monoculture_Success=False)
        if COMPUTE_MEASURABLE_CUE and row["Community"] == 1:
            row.update(measurable_cue_monoculture(
                u1[index:index + 1], l1[index:index + 1], MAINTENANCE_COST,
                rho1, omega1, C0_1[index], R0_1, T_SPAN))

    return species_data


# =========================
# 5. Main program
# =========================
def simulate_indexed(task):
    index, seed = task
    return index, simulate(seed)


def main():
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()
    if not seeds:
        raise ValueError("N_SIMULATIONS must be at least 1")

    workers = max(1, min(N_WORKERS, cpu_count(), len(seeds)))
    started = perf_counter()
    completed = 0
    all_species_data_nested = [None] * len(seeds)
    print(
        f"Starting {len(seeds)} simulations with {workers} worker(s); "
        f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}.",
        flush=True
    )
    print(f"Monoculture CUE assays: {'enabled (Community 1)' if COMPUTE_MEASURABLE_CUE else 'disabled'}.", flush=True)
    print("Progress is reported after each seed and every 30 seconds while waiting.", flush=True)

    with Pool(workers) as pool:
        results = pool.imap_unordered(simulate_indexed, enumerate(seeds), chunksize=1)
        while completed < len(seeds):
            try:
                index, species_data = results.next(timeout=30)
            except PoolTimeoutError:
                print(
                    f"Still computing: {completed}/{len(seeds)} seeds complete; "
                    f"elapsed {(perf_counter() - started) / 60:.1f} min.",
                    flush=True
                )
                continue
            # Keep CSV row order identical to the original seed order.
            all_species_data_nested[index] = species_data
            completed += 1
            print(
                f"Completed {completed}/{len(seeds)} seeds "
                f"(seed={seeds[index]}); elapsed {(perf_counter() - started) / 60:.1f} min.",
                flush=True
            )

    all_species_data = [
        row
        for one_seed_result in all_species_data_nested
        if one_seed_result
        for row in one_seed_result
    ]

    df = pd.DataFrame(all_species_data)
    df.to_csv(os.path.join(code_path, "coal.csv"), index=False)

    community_status = df[["Seed", "Community", "Stability_Status"]].drop_duplicates(
        ["Seed", "Community"]
    )
    # Always write the file so disabling assays cannot leave stale measurements.
    rmax_data = df.loc[df["Community"] == 1, [
        "Seed", "Community", "Species_ID", "Species_CUE", "Abundance", "rmax", "t_star",
        "growth_CUE", "Monoculture_End_Time", "Monoculture_Success",
    ]].rename(columns={"Species_CUE": "intrinsic_CUE"})
    rmax_data.to_csv(os.path.join(code_path, "rmax_cue.csv"), index=False)


    summary_df = (
        df.groupby("Community")
        .agg(
            Mean_Community_CUE=("Community_CUE", "mean"),
            Mean_Community_CUE_surv=("Community_CUE_surv", "mean"),
            Mean_Abundance=("Abundance", "mean"),
            Mean_Theory_Abundance=("Theory_Abundance", "mean"),
            Mean_Competition=("Competition", "mean"),
            Mean_Heterospecific_Competition_Pressure=("Heterospecific_Competition_Pressure", "mean"),
            Mean_Facilitation=("Facilitation", "mean"),
            Mean_Depletion=("Depletion", "mean"),
            Mean_UptakeVar=("UptakeVar", "mean")
        )
        .reset_index()
    )

    params_rows = []
    for comm in sorted(df["Community"].astype(str).unique()):
        dat_comm = df[df["Community"].astype(str) == comm].copy()
        params = estimate_theory_params_mechanistic(dat_comm, survival_threshold=SURVIVAL_THRESHOLD)

        if params is None:
            params_rows.append({
                "Community": comm,
                "eps_c": np.nan,
                "chi_bar": np.nan,
                "U_bar": np.nan,
                "H": np.nan,
                "Cmax": np.nan,
                "Theory_R2_log": np.nan,
                "N_total": len(dat_comm),
                "N_survivors": int(np.sum(dat_comm["Abundance"] > SURVIVAL_THRESHOLD)),
                "N_seeds": dat_comm["Seed"].nunique(),
                "NearThresholdUsed_median": np.nan
            })
        else:
            params_rows.append({
                "Community": comm,
                "eps_c": params["eps_c"],
                "chi_bar": params["chi_bar"],
                "U_bar": params["U_bar"],
                "H": params["H"],
                "Cmax": params["Cmax"],
                "Theory_R2_log": params["Theory_R2_log"],
                "N_total": params["N_total"],
                "N_survivors": params["N_survivors"],
                "N_seeds": params["N_seeds"],
                "NearThresholdUsed_median": params["NearThresholdUsed_median"]
            })

    params_df = pd.DataFrame(params_rows)

    summary_df.to_csv(os.path.join(code_path, "coal_summary.csv"), index=False)
    params_df.to_csv(os.path.join(code_path, "cue_abundance_theory_params.csv"), index=False)

    print("Simulation completed. Results saved:")
    print(os.path.join(code_path, "coal.csv"))
    print(os.path.join(code_path, "coal_summary.csv"))
    print(os.path.join(code_path, "cue_abundance_theory_params.csv"))

    print(os.path.join(code_path, "rmax_cue.csv"))
    print("\nStability status counts:")
    print(community_status.groupby(["Community", "Stability_Status"]).size())

    print("\nSummary results by community:")
    print(summary_df)

    print("\nTheoretical parameter results:")
    print(params_df)


if __name__ == "__main__":
    main()
