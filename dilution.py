from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp


# this script simulates rare species invasion in microbial communities. 



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


def choose_resources_for_second_community(M_pool, M1, M2, resource_indices1, rng):
    """Select resource indices for the second community."""
    if M1 > M2:
        return rng.choice(resource_indices1, M2, replace=False)
    if M1 < M2:
        remaining_resources = np.setdiff1d(np.arange(M_pool), resource_indices1)
        additional_resources = rng.choice(remaining_resources, M2 - M1, replace=False)
        return np.concatenate([resource_indices1, additional_resources])
    return resource_indices1.copy()


# =========================
# simulation parameters and main function
# =========================
BASE_SEED = 50
N_SIMULATIONS = 10

# Save results in the same folder as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RARE_FILE = os.path.join(SCRIPT_DIR, "rare.csv")
SUMMARY_FILE = os.path.join(SCRIPT_DIR, "rare_summary.csv")

# Species pool and resource pool parameters
N_POOL = 1000
M_POOL = 100
N_MODULES = 1
S_RATIO = 1.0
LEAKAGE_RATE = 0.2

# Community parameters
N1, M1 = 100, 50
N2, M2 = 100, 50

# Physiological parameters
MAINTENANCE_COST = 0.2
RHO_VALUE = 0.6
OMEGA_VALUE = 0.1
T_SPAN = (0, 50000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Dilution rates for invasion experiment
DILUTION_RATES = [0.01, 0.1]


def simulate(args):
    """Simulate rare species invasion with specified dilution rate."""
    seed, dilution_rate = args
    rng = np.random.default_rng(seed)

    # Generate species and resource pools
    u_pool = modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # Community 1
    species_indices1 = rng.choice(N_POOL, N1, replace=False)
    resource_indices1 = rng.choice(M_POOL, M1, replace=False)

    u1 = u_pool[np.ix_(species_indices1, resource_indices1)]
    l1 = l_pool[np.ix_(species_indices1, resource_indices1, resource_indices1)]

    lambda_alpha1 = np.full(M1, LEAKAGE_RATE)
    rho1 = np.full(M1, RHO_VALUE)
    omega1 = np.full(M1, OMEGA_VALUE)
    C0_1 = np.full(N1, C0_VALUE)
    R0_1 = np.full(M1, R0_VALUE)

    sol1 = solve_micrm(
        N1, M1, u1, l1, MAINTENANCE_COST, lambda_alpha1,
        rho1, omega1, C0_1, R0_1, T_SPAN
    )

    # Community 2
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

    sol2 = solve_micrm(
        N2, M2, u2, l2, MAINTENANCE_COST, lambda_alpha2,
        rho2, omega2, C0_2, R0_2, T_SPAN
    )

    # Coalesced community
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = resource_indices1 if M1 >= M2 else resource_indices2

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)]
    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]

    N3 = N1 + N2
    M3 = len(resource_indices3)
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)

    # Apply dilution to invader community
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1] * dilution_rate])
    R0_3 = np.full(M3, R0_VALUE)

    sol3 = solve_micrm(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3,
        rho3, omega3, C0_3, R0_3, T_SPAN
    )

    # Final abundances
    C_final1 = sol1.y[:N1, -1]
    C_final2 = sol2.y[:N2, -1]
    C_final3 = sol3.y[:N3, -1]

    # CUE calculations
    species_CUE1 = compute_species_CUE(u1, R0_1, LEAKAGE_RATE, MAINTENANCE_COST)
    species_CUE2 = compute_species_CUE(u2, R0_2, LEAKAGE_RATE, MAINTENANCE_COST)
    species_CUE3 = compute_species_CUE(u3, R0_3, LEAKAGE_RATE, MAINTENANCE_COST)

    # Survivors
    survivors1 = np.where(C_final1 > SURVIVAL_THRESHOLD)[0]
    survivors2 = np.where(C_final2 > SURVIVAL_THRESHOLD)[0]
    survivors3 = np.where(C_final3 > SURVIVAL_THRESHOLD)[0]

    community_CUE1 = safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2 = safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3 = safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])

    # Competition metrics
    competition_comm1 = community_level_competition(u1)
    competition_comm2 = community_level_competition(u2)
    competition_comm3 = community_level_competition(u3)

    competition_species1 = species_level_competition(u1)
    competition_species2 = species_level_competition(u2)
    competition_species3 = species_level_competition(u3)

    competition_dot1 = species_level_competition_dot(u1)
    competition_dot2 = species_level_competition_dot(u2)
    competition_dot3 = species_level_competition_dot(u3)

    # Facilitation metrics
    L_eff1 = calculate_effective_leakage(u1, l1)
    L_eff2 = calculate_effective_leakage(u2, l2)
    L_eff3 = calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(L_eff1, axis=1)
    facilitation2 = np.mean(L_eff2, axis=1)
    facilitation3 = np.mean(L_eff3, axis=1)

    # Uptake variance
    uptake_var1 = compute_uptake_variance(u1)
    uptake_var2 = compute_uptake_variance(u2)
    uptake_var3 = compute_uptake_variance(u3)

    # Resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    # Total abundance
    total_abundance1 = np.sum(C_final1)
    total_abundance2 = np.sum(C_final2)
    total_abundance3 = np.sum(C_final3)

    # Dominance in merged community
    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community1" if origin1_in_coalesced > origin2_in_coalesced else "Community2"

    species_data = []

    # Community 1 data
    for i in range(N1):
        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 1,
            "Species_ID": i + 1,
            "Origin": "Comm1",
            "Species_CUE": species_CUE1[i],
            "Community_CUE": community_CUE1,
            "Abundance": C_final1[i],
            "Total_Abundance": total_abundance1,
            "Dominant_Community": dominant,
            "Competition": competition_comm1,
            "Species_Competition": competition_species1[i],
            "Species_Competition_Dot": competition_dot1[i],
            "Facilitation": facilitation1[i],
            "Depletion": depletion1,
            "UptakeVar": uptake_var1[i],
            "Species_Index": int(species_indices1[i])
        })

    # Community 2 data
    for i in range(N2):
        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 2,
            "Species_ID": i + 1,
            "Origin": "Comm2",
            "Species_CUE": species_CUE2[i],
            "Community_CUE": community_CUE2,
            "Abundance": C_final2[i],
            "Total_Abundance": total_abundance2,
            "Dominant_Community": dominant,
            "Competition": competition_comm2,
            "Species_Competition": competition_species2[i],
            "Species_Competition_Dot": competition_dot2[i],
            "Facilitation": facilitation2[i],
            "Depletion": depletion2,
            "UptakeVar": uptake_var2[i],
            "Species_Index": int(species_indices2[i])
        })

    # Community 3 data
    for i in range(N3):
        origin = "Comm1" if i < N1 else "Comm2"
        species_index = int(species_indices1[i]) if i < N1 else int(species_indices2[i - N1])

        species_data.append({
            "Seed": seed,
            "DilutionRate": dilution_rate,
            "Community": 3,
            "Species_ID": i + 1,
            "Origin": origin,
            "Species_CUE": species_CUE3[i],
            "Community_CUE": community_CUE3,
            "Abundance": C_final3[i],
            "Total_Abundance": total_abundance3,
            "Dominant_Community": dominant,
            "Competition": competition_comm3,
            "Species_Competition": competition_species3[i],
            "Species_Competition_Dot": competition_dot3[i],
            "Facilitation": facilitation3[i],
            "Depletion": depletion3,
            "UptakeVar": uptake_var3[i],
            "Species_Index": species_index
        })

    return species_data


def main():
    """Main function to run rare species invasion simulations."""
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    param_list = [(seed, dr) for seed in seeds for dr in DILUTION_RATES]

    print("Starting rare species invasion simulations...")
    print(f"  Number of seeds: {len(seeds)}")
    print(f"  Dilution rates: {DILUTION_RATES}")
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
    df.to_csv(RARE_FILE, index=False)

    print("\nSimulation completed!")
    print(f"Detailed results saved to: {RARE_FILE}")


if __name__ == "__main__":
    main()