"""
Plot competition and facilitation metrics as a function of resource inflow rate (RHO).
This script analyzes how species competition and facilitation change with different 
resource availability levels.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings("ignore")

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RHO_FILE = os.path.join(SCRIPT_DIR, "rho_resource.csv")
OUTPUT_DIR = SCRIPT_DIR

# Survival threshold
SURVIVAL_THRESHOLD = 1e-5

# Plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 0.4,
})


def load_and_filter_data(file_path):
    """Load data and filter for surviving species."""
    df = pd.read_csv(file_path)
    df_survivors = df[df["Abundance"] > SURVIVAL_THRESHOLD].copy()
    return df_survivors


def calculate_mean_metrics(df):
    """Calculate mean competition and facilitation for each RHO."""
    metrics = df.groupby("RHO").agg({
        "Species_Competition": ["mean", "std", "sem"],
        "Species_Competition_Dot": ["mean", "std", "sem"],
        "Facilitation": ["mean", "std", "sem"],
        "Seed": "count"  # Number of data points
    }).reset_index()
    
    # Flatten column names
    metrics.columns = ["_".join(col).strip("_") for col in metrics.columns.values]
    metrics.rename(columns={
        "RHO": "RHO",
        "Species_Competition_mean": "Competition_Mean",
        "Species_Competition_std": "Competition_Std",
        "Species_Competition_sem": "Competition_SEM",
        "Species_Competition_Dot_mean": "Competition_Dot_Mean",
        "Species_Competition_Dot_std": "Competition_Dot_Std",
        "Species_Competition_Dot_sem": "Competition_Dot_SEM",
        "Facilitation_mean": "Facilitation_Mean",
        "Facilitation_std": "Facilitation_Std",
        "Facilitation_sem": "Facilitation_SEM",
        "Seed_count": "N_Points"
    }, inplace=True)
    
    return metrics


def plot_competition_vs_rho(metrics, output_path):
    """Plot species competition as a function of RHO."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        metrics["RHO"],
        metrics["Competition_Mean"],
        yerr=metrics["Competition_SEM"],
        marker="o",
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color="#2F4858",
        alpha=0.8,
        label="Species Competition"
    )
    
    ax.set_xlabel(r"Resource Inflow Rate ($\rho$)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Species Competition (Cosine Similarity)", fontsize=16, fontweight="bold")
    ax.set_title("Competition vs Resource Availability", fontsize=18, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(metrics["RHO"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_facilitation_vs_rho(metrics, output_path):
    """Plot facilitation as a function of RHO."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        metrics["RHO"],
        metrics["Facilitation_Mean"],
        yerr=metrics["Facilitation_SEM"],
        marker="s",
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color="#A8C3A6",
        alpha=0.8,
        label="Facilitation"
    )
    
    ax.set_xlabel(r"Resource Inflow Rate ($\rho$)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Facilitation (Effective Leakage)", fontsize=16, fontweight="bold")
    ax.set_title("Facilitation vs Resource Availability", fontsize=18, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(metrics["RHO"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_metrics(metrics, output_path):
    """Plot both competition and facilitation in a 2-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Competition
    ax1 = axes[0]
    ax1.errorbar(
        metrics["RHO"],
        metrics["Competition_Mean"],
        yerr=metrics["Competition_SEM"],
        marker="o",
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color="#2F4858",
        alpha=0.8
    )
    
    ax1.set_xlabel(r"Resource Inflow Rate ($\rho$)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Species Competition", fontsize=16, fontweight="bold")
    ax1.set_title("(A) Competition vs Resource Availability", fontsize=18, fontweight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xticks(metrics["RHO"])
    
    # Panel B: Facilitation
    ax2 = axes[1]
    ax2.errorbar(
        metrics["RHO"],
        metrics["Facilitation_Mean"],
        yerr=metrics["Facilitation_SEM"],
        marker="s",
        markersize=10,
        linewidth=2.5,
        capsize=6,
        capthick=2,
        color="#A8C3A6",
        alpha=0.8
    )
    
    ax2.set_xlabel(r"Resource Inflow Rate ($\rho$)", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Facilitation (Effective Leakage)", fontsize=16, fontweight="bold")
    ax2.set_title("(B) Facilitation vs Resource Availability", fontsize=18, fontweight="bold", pad=15)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks(metrics["RHO"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(metrics):
    """Print summary statistics for each RHO value."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Competition and Facilitation vs RHO")
    print("="*80)
    print("\nResource Inflow Analysis:")
    print("-" * 80)
    
    for _, row in metrics.iterrows():
        print(f"\nRHO = {row['RHO']:.1f}:")
        print(f"  Competition: {row['Competition_Mean']:.4f} ± {row['Competition_SEM']:.4f} (n={int(row['N_Points'])})")
        print(f"  Facilitation: {row['Facilitation_Mean']:.4f} ± {row['Facilitation_SEM']:.4f}")
    
    print("\n" + "="*80)


def main():
    """Main function to analyze and plot RHO effects."""
    print("Loading data from:", RHO_FILE)
    
    if not os.path.exists(RHO_FILE):
        print(f"Error: {RHO_FILE} not found!")
        print("Please run rho_analysis.py first to generate the data.")
        return
    
    # Load and filter data
    df = load_and_filter_data(RHO_FILE)
    print(f"Loaded {len(df)} surviving species records")
    
    # Check if RHO column exists
    if "RHO" not in df.columns:
        print("Error: 'RHO' column not found in data!")
        print("Please re-run rho_analysis.py with the updated version.")
        return
    
    # Calculate metrics
    metrics = calculate_mean_metrics(df)
    
    # Print summary statistics
    print_summary_statistics(metrics)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_competition_vs_rho(
        metrics,
        os.path.join(OUTPUT_DIR, "competition_vs_rho.png")
    )
    plot_facilitation_vs_rho(
        metrics,
        os.path.join(OUTPUT_DIR, "facilitation_vs_rho.png")
    )
    plot_combined_metrics(
        metrics,
        os.path.join(OUTPUT_DIR, "competition_facilitation_vs_rho.png")
    )
    
    # Save metrics to CSV
    metrics_file = os.path.join(OUTPUT_DIR, "rho_metrics_summary.csv")
    metrics.to_csv(metrics_file, index=False)
    print(f"\nMetrics summary saved to: {metrics_file}")
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
