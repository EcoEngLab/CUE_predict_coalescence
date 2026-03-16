import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from scipy.integrate import solve_ivp


# this script simulates community coalescence with varying degrees of resource overlap between two communities.


# =========================
# functions for generating modular uptake and leakage matrices, solving MiCRM, and calculating metrics
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

        uptake = u * (R * (1 - lambda_alpha))
        dCdt = C * (np.sum(uptake, axis=1) - m)

        dRdt = rho - omega * R
        consumption = np.sum(C[:, None] * u * R, axis=0)
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


def calculate_effective_leakage(u, l):
    """Calculate effective leakage for each species."""
    return np.einsum("ia,iab->ib", u, l)


# =========================
# main simulation parameters and execution
# =========================
BASE_SEED = 50
N_SIMULATIONS = 50

# Save result CSV in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COAL_RESOURCE_FILE = os.path.join(SCRIPT_DIR, "coal_resource.csv")

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
T_SPAN = (0, 100000)

# Initial conditions
C0_VALUE = 0.01
R0_VALUE = 1.0

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Resource overlap ratios for experiment
OVERLAP_RATIOS = [0.25, 0.5, 0.75]


def simulate_overlap(args):
    """Simulate community coalescence with specified resource overlap ratio."""
    seed, overlap_ratio = args
    rng = np.random.default_rng(seed)

    # Generate species and resource pools
    u_pool = modular_uptake(N_POOL, M_POOL, N_MODULES, S_RATIO, rng)
    l_pool = generate_l_tensor(N_POOL, M_POOL, N_MODULES, S_RATIO, LEAKAGE_RATE, u_pool, rng)

    # Calculate shared and unique resources
    overlap_n = int(M1 * overlap_ratio)
    unique_n = M1 - overlap_n

    all_resources = np.arange(M_POOL)
    overlap_resources = rng.choice(all_resources, overlap_n, replace=False)
    remain_resources = np.setdiff1d(all_resources, overlap_resources)
    res1_unique = rng.choice(remain_resources, unique_n, replace=False)
    res2_unique = rng.choice(np.setdiff1d(remain_resources, res1_unique), unique_n, replace=False)

    resource_indices1 = np.concatenate([overlap_resources, res1_unique])
    resource_indices2 = np.concatenate([overlap_resources, res2_unique])

    # Community 1
    species_indices1 = rng.choice(N_POOL, N1, replace=False)

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

    # Community 3
    species_indices3 = np.concatenate([species_indices1, species_indices2])
    resource_indices3 = np.union1d(resource_indices1, resource_indices2)
    N3 = N1 + N2
    M3 = len(resource_indices3)

    u3 = u_pool[np.ix_(species_indices3, resource_indices3)].copy()

    # Restrict species to only utilize resources from their original community
    mask1 = np.isin(resource_indices3, resource_indices1)
    mask2 = np.isin(resource_indices3, resource_indices2)
    u3[:N1, ~mask1] = 0.0
    u3[N1:, ~mask2] = 0.0

    l3 = l_pool[np.ix_(species_indices3, resource_indices3, resource_indices3)]
    lambda_alpha3 = np.full(M3, LEAKAGE_RATE)
    rho3 = np.full(M3, RHO_VALUE)
    omega3 = np.full(M3, OMEGA_VALUE)
    C0_3 = np.concatenate([sol1.y[:N1, -1], sol2.y[:N2, -1]])
    R0_3 = np.full(M3, R0_VALUE)

    sol3 = solve_micrm(
        N3, M3, u3, l3, MAINTENANCE_COST, lambda_alpha3,
        rho3, omega3, C0_3, R0_3, T_SPAN
    )

    # Extract community metrics
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

    # Community CUE
    community_CUE1 = safe_weighted_average(species_CUE1[survivors1], C_final1[survivors1])
    community_CUE2 = safe_weighted_average(species_CUE2[survivors2], C_final2[survivors2])
    community_CUE3 = safe_weighted_average(species_CUE3[survivors3], C_final3[survivors3])

    # Competition metrics
    competition_comm1 = community_level_competition(u1)
    competition_comm2 = community_level_competition(u2)
    competition_comm3 = community_level_competition(u3)

    # Facilitation metrics
    L_eff1 = calculate_effective_leakage(u1, l1)
    L_eff2 = calculate_effective_leakage(u2, l2)
    L_eff3 = calculate_effective_leakage(u3, l3)

    facilitation1 = np.mean(np.sum(L_eff1, axis=1))
    facilitation2 = np.mean(np.sum(L_eff2, axis=1))
    facilitation3 = np.mean(np.sum(L_eff3, axis=1))

    # Resource depletion
    depletion1 = np.sum(sol1.y[N1:, -1])
    depletion2 = np.sum(sol2.y[N2:, -1])
    depletion3 = np.sum(sol3.y[N3:, -1])

    # Total abundance
    total_abundance_1 = np.sum(C_final1)
    total_abundance_2 = np.sum(C_final2)
    total_abundance_3 = np.sum(C_final3)

    # Dominance in merged community
    origin1_in_coalesced = np.sum(C_final3[:N1])
    origin2_in_coalesced = np.sum(C_final3[N1:])
    dominant = "Community1" if origin1_in_coalesced > origin2_in_coalesced else "Community2"

    # Bray-Curtis similarity
    parent1_vec = np.concatenate([C_final1, np.zeros(N2)])
    parent2_vec = np.concatenate([np.zeros(N1), C_final2])
    coalesced_vec = C_final3

    bc_diss_3vs1 = (
        np.sum(np.abs(coalesced_vec - parent1_vec)) / np.sum(coalesced_vec + parent1_vec)
        if np.sum(coalesced_vec + parent1_vec) > 0 else 1.0
    )
    bc_diss_3vs2 = (
        np.sum(np.abs(coalesced_vec - parent2_vec)) / np.sum(coalesced_vec + parent2_vec)
        if np.sum(coalesced_vec + parent2_vec) > 0 else 1.0
    )
    sim_3vs1 = 1 - bc_diss_3vs1
    sim_3vs2 = 1 - bc_diss_3vs2

    return {
        "Seed": seed,
        "Overlap": overlap_ratio,
        "CUE1": community_CUE1,
        "CUE2": community_CUE2,
        "CUE3": community_CUE3,
        "Num_Survivors1": len(survivors1),
        "Num_Survivors2": len(survivors2),
        "Num_Survivors3": len(survivors3),
        "Competition1": competition_comm1,
        "Competition2": competition_comm2,
        "Competition3": competition_comm3,
        "Facilitation1": facilitation1,
        "Facilitation2": facilitation2,
        "Facilitation3": facilitation3,
        "Depletion1": depletion1,
        "Depletion2": depletion2,
        "Depletion3": depletion3,
        "Total_Abundance_1": total_abundance_1,
        "Total_Abundance_2": total_abundance_2,
        "Total_Abundance_3": total_abundance_3,
        "Dominant_Community": dominant,
        "Similarity_3vs1": sim_3vs1,
        "Similarity_3vs2": sim_3vs2
    }


def main():
    """Main function to run resource overlap coalescence simulations."""
    seed_generator = np.random.default_rng(BASE_SEED)
    seeds = seed_generator.integers(0, 2**32 - 1, size=N_SIMULATIONS, dtype=np.uint32).tolist()

    args_list = [(seed, overlap) for seed in seeds for overlap in OVERLAP_RATIOS]

    print("Starting resource overlap coalescence simulations...")
    print(f"  Number of seeds: {len(seeds)}")
    print(f"  Overlap ratios: {OVERLAP_RATIOS}")
    print(f"  Total simulations: {len(args_list)}")
    print(f"  CPU cores: {cpu_count()}")

    with Pool(cpu_count()) as pool:
        results = pool.map(simulate_overlap, args_list)

    df = pd.DataFrame(results)
    df.to_csv(COAL_RESOURCE_FILE, index=False)

    print("\nSimulation completed!")
    print(f"  Results saved to: {COAL_RESOURCE_FILE}")
    print(f"  Total records: {len(df)}")


if __name__ == "__main__":
    main()