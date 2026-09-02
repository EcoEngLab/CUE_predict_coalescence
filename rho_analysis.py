from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp


# This script analyzes the effect of resource inflow rate (RHO) on species competition and facilitation.


# =========================
# functions for generating matrices, solving MiCRM, and calculating metrics
# =========================
def modular_uptake(N, M, N_modules, s_ratio, rng):
    """Generate a modular uptake matrix u."""
    assert N_modules <= M and N_modules <= N, "N_modules must be <= both M and N"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    sC = N // N_modules
    dC = N - (N_modules * sC)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [
        list(range(x - 1, y))
        for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))
    ]

    diffC = np.full(N_modules, sC, dtype=int)
    if dC > 0:
        diffC[rng.choice(N_modules, dC, replace=False)] += 1
    mC = [
        list(range(x - 1, y))
        for x, y in zip((np.cumsum(diffC) - diffC + 1), np.cumsum(diffC))
    ]

    u = rng.random((N, M))

    for x, y in zip(mC, mR):
        u[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(u, axis=1, keepdims=True)
    u = u / row_sums
    return u


def modular_leakage(M, N_modules, s_ratio, lam, rng):
    """Generate a modular leakage matrix l."""
    assert N_modules <= M, "N_modules must be <= M"

    sR = M // N_modules
    dR = M - (N_modules * sR)

    diffR = np.full(N_modules, sR, dtype=int)
    if dR > 0:
        diffR[rng.choice(N_modules, dR, replace=False)] += 1
    mR = [
        list(range(x - 1, y))
        for x, y in zip((np.cumsum(diffR) - diffR + 1), np.cumsum(diffR))
    ]

    l = rng.random((M, M))

    for i, x in enumerate(mR):
        for j, y in enumerate(mR):
            if i == j or i + 1 == j:
                l[np.ix_(x, y)] *= s_ratio

    row_sums = np.sum(l, axis=1, keepdims=True)
    l = lam * l / row_sums
    return l


def generate_l_tensor(N, M, N_modules, s_ratio, lam, u, rng):
    """Generate a 3D leakage tensor l."""
    l_tensor = np.zeros((N, M, M))
    for i in range(N):
        l_tensor[i] = modular_leakage(M, N_modules, s_ratio, lam, rng)
    return l_tensor


def safe_weighted_average(values, weights):
    """Compute a weighted average safely."""
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan
    return np.sum(values * weights) / total_weight


def compute_species_CUE(u, R_ref, lam, m):
    """Compute species-level CUE."""
    total_uptake = np.sum(u * R_ref, axis=1)
    net_uptake = np.sum(u * R_ref * (1 - lam), axis=1) - m
    species_CUE = net_uptake / (total_uptake + 1e-12)
    return species_CUE


def solve_micrm(
    N, M, u, l, m, lambda_alpha, rho, omega, C0, R0,
    t_span, t_eval=None, tol=1e-5, method="BDF"
):
    """Solve the MiCRM model using solve_ivp with an event to detect equilibrium."""
    def dCdt_Rdt(t, y):
        C = y[:N]
        R = y[N:]

        uptake = u * (R * (1 - lambda_alpha))              # (N, M)
        dCdt = C * (np.sum(uptake, axis=1) - m)

        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)   # (M,)
        dRdt -= consumption

        leakage = np.einsum("i,j,ij,ijk->k", C, R, u, l)
        dRdt += leakage

        return np.concatenate([dCdt, dRdt])

    def equilibrium_event(t, y):
        deriv = dCdt_Rdt(t, y)
        return np.max(np.abs(deriv)) - tol

    equilibrium_event.terminal = True
    equilibrium_event.direction = -1

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 100)

    Y0 = np.concatenate([C0, R0])

    sol = solve_ivp(
        dCdt_Rdt,
        t_span,
        Y0,
        t_eval=t_eval,
        method=method,
        events=equilibrium_event
    )
    return sol


def calculate_effective_leakage(u, l):
    """Calculate effective leakage for each species."""
    return np.einsum("ia,iab->ib", u, l)


def community_level_competition(u):
    """Calculate community-level competition based on average cosine similarity."""
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
    """Calculate species-level competition based on average cosine similarity to others."""
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
    """Calculate species-level competition based on dot product."""
    N, _ = u.shape
    if N < 2:
        return np.full(N, np.nan)

    comp_matrix = u @ u.T
    np.fill_diagonal(comp_matrix, 0.0)
    comp = np.sum(comp_matrix, axis=1) / (N - 1)
    return comp


def compute_uptake_variance(u):
    """Calculate the variance of uptake for each species across resources."""
    return np.var(u, axis=1)


# =========================
# simulation parameters and main function
# =========================
BASE_SEED = 100
N_SIMULATIONS = 10

# Save results in the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RHO_FILE = os.path.join(SCRIPT_DIR, "rho_resource.csv")

# Species pool and resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1.0
LEAKAGE_RATE = 0.2

# Community parameters
N_COMM = 100
M_COMM = 50

# Physiological parameters
MAINTENANCE_COST = 0.2
OMEGA_VALUE = 0.1
T_SPAN = (0, 50000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Resource inflow rates for analysis
RHO_VALUES = [0.4, 0.6, 0.8]


def simulate(args):
    """Simulate community dynamics with specified RHO value."""
    seed, rho_value = args
    rng = np.random.default_rng(seed)

    # Generate species and resource pools
    u_pool = modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # Select species and resources for community
    species_indices = rng.choice(N_POOL, N_COMM, replace=False)
    resource_indices = rng.choice(M_POOL, M_COMM, replace=False)

    u = u_pool[np.ix_(species_indices, resource_indices)]
    l = l_pool[np.ix_(species_indices, resource_indices, resource_indices)]

    lambda_alpha = np.full(M_COMM, LEAKAGE_RATE)
    rho = np.full(M_COMM, rho_value)
    omega = np.full(M_COMM, OMEGA_VALUE)
    C0 = np.full(N_COMM, C0_VALUE)
    R0 = np.full(M_COMM, R0_VALUE)

    sol = solve_micrm(
        N_COMM, M_COMM, u, l, MAINTENANCE_COST, lambda_alpha,
        rho, omega, C0, R0, T_SPAN
    )

    # Final abundances and resources
    C_final = sol.y[:N_COMM, -1]
    R_final = sol.y[N_COMM:, -1]

    # CUE calculations
    species_CUE = compute_species_CUE(u, R0, LEAKAGE_RATE, MAINTENANCE_COST)

    # Survivors
    survivors = np.where(C_final > SURVIVAL_THRESHOLD)[0]
    community_CUE = safe_weighted_average(species_CUE[survivors], C_final[survivors])

    # Competition metrics
    competition_comm = community_level_competition(u)
    competition_species = species_level_competition(u)
    competition_dot = species_level_competition_dot(u)

    # Facilitation metrics
    L_eff = calculate_effective_leakage(u, l)
    facilitation = np.mean(L_eff, axis=1)

    # Uptake variance
    uptake_var = compute_uptake_variance(u)

    # Resource depletion
    total_resource = np.sum(R_final)
    
    # Total abundance
    total_abundance = np.sum(C_final)

    # Collect species data
    species_data = []
    for i in range(N_COMM):
        species_data.append({
            "Seed": seed,
            "RHO": rho_value,
            "Species_ID": i + 1,
            "Species_CUE": species_CUE[i],
            "Community_CUE": community_CUE,
            "Abundance": C_final[i],
            "Total_Abundance": total_abundance,
            "Community_Competition": competition_comm,
            "Species_Competition": competition_species[i],
            "Species_Competition_Dot": competition_dot[i],
            "Facilitation": facilitation[i],
            "Total_Resource": total_resource,
            "UptakeVar": uptake_var[i],
            "Species_Index": int(species_indices[i]),
            "N_Survivors": len(survivors)
        })

    return species_data


def main():
    """Main function to run RHO analysis simulations."""
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    param_list = [(seed, rho) for seed in seeds for rho in RHO_VALUES]

    print("Starting RHO analysis simulations...")
    print(f"  Number of seeds: {len(seeds)}")
    print(f"  RHO values: {RHO_VALUES}")
    print(f"  Total simulations: {len(param_list)}")
    print(f"  CPU cores: {cpu_count()}")

    with Pool(cpu_count()) as pool:
        all_data_nested = pool.map(simulate, param_list)

    all_data = [
        row
        for result in all_data_nested
        if result
        for row in result
    ]

    df = pd.DataFrame(all_data)
    df.to_csv(RHO_FILE, index=False)

    print("\nSimulation completed!")
    print(f"Results saved to: {RHO_FILE}")


if __name__ == "__main__":
    main()
