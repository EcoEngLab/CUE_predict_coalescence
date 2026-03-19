# ====================================================================================================
# ======================================== settings =========================================
# ====================================================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

SURVIVAL_THRESHOLD = 1e-5

pal_rgb = {
    "1": "#D8A39A",
    "2": "#A8C3A6",
    "3": "#9FB7CC"
}

theory_line_color = "#2F4858"

community_labels = {
    "1": "Community 1",
    "2": "Community 2",
    "3": "Coalesced Community"
}

xlabels = {
    "1": "Species-level CUE of Community 1",
    "2": "Species-level CUE of Community 2",
    "3": "Species-level CUE of Coalesced Community"
}

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
    "legend.fontsize": 14,
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def style_ax(ax, grid=False):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)

    ax.tick_params(axis="both", width=0.4, colors="black", pad=4)

    if grid:
        ax.grid(True, which="major", color="#E5E5E5", linewidth=0.35)
        ax.grid(True, which="minor", color="#F2F2F2", linewidth=0.2)
    else:
        ax.grid(False)

def first_unique(series):
    vals = pd.Series(series).dropna().unique()
    return vals[0] if len(vals) > 0 else np.nan

df = pd.read_csv("coal.csv")
df = df.rename(columns={"Species_Competition_Dot": "Species_Competition2"})
df["Community"] = df["Community"].astype(str)
df["Species_ID"] = pd.to_numeric(df["Species_ID"], errors="coerce")
df["Species_CUE"] = pd.to_numeric(df["Species_CUE"], errors="coerce")
df["Abundance"] = pd.to_numeric(df["Abundance"], errors="coerce")

df_surv = df[df["Abundance"] > SURVIVAL_THRESHOLD].copy()
df_surv["log10_Abundance"] = np.log10(df_surv["Abundance"])

params_df = pd.read_csv("cue_abundance_theory_params.csv")
params_df["Community"] = params_df["Community"].astype(str)

df_resource = pd.read_csv("coal_resource.csv")
df_resource = df_resource.rename(columns={
    "Similarity_3vs1": "Sim_3vs1",
    "Similarity_3vs2": "Sim_3vs2"
})
df_resource["Overlap"] = df_resource["Overlap"].astype(str)

df_rare = pd.read_csv("rare.csv")
df_rare = df_rare.rename(columns={
    "Abundance": "C_final",
    "Species_CUE": "CUE"
})




# ====================================================================================================
# ================================ Species CUE vs Abundance + Theory =============================
# ====================================================================================================

y_min = np.nanmin(df_surv["log10_Abundance"])
y_max = np.nanmax(df_surv["log10_Abundance"])

fig = plt.figure(figsize=(8.6, 12))
gs = fig.add_gridspec(
    3, 2,
    width_ratios=[1.45, 3.55],
    hspace=0.38,
    wspace=0.06
)

for i, comm in enumerate(["1", "2", "3"]):
    dat_surv = df_surv[df_surv["Community"] == comm].copy()
    dat_surv = dat_surv[
        np.isfinite(dat_surv["Species_CUE"]) &
        np.isfinite(dat_surv["log10_Abundance"])
    ]

    dat_full = df[df["Community"] == comm].copy()
    dat_full = dat_full[
        np.isfinite(dat_full["Species_CUE"]) &
        np.isfinite(dat_full["Abundance"])
    ]

    row = params_df[params_df["Community"] == comm].iloc[0]
    eps_c = pd.to_numeric(row["eps_c"], errors="coerce")
    H = pd.to_numeric(row["H"], errors="coerce")
    Cmax = pd.to_numeric(row["Cmax"], errors="coerce")

    ax_hist = fig.add_subplot(gs[i, 0])
    ax_main = fig.add_subplot(gs[i, 1], sharey=ax_hist)

    ax_hist.hist(
        dat_surv["log10_Abundance"].dropna(),
        bins=50,
        orientation="horizontal",
        color=pal_rgb[comm],
        alpha=0.45,
        edgecolor="black",
        linewidth=0.3
    )
    ax_hist.set_ylim(y_min, y_max)
    ax_hist.invert_xaxis()
    ax_hist.set_xlabel("Density")
    ax_hist.set_ylabel(r"Abundance ($\log_{10}$ scale)", labelpad=10)
    style_ax(ax_hist, grid=False)

    ax_main.scatter(
        dat_surv["Species_CUE"],
        dat_surv["log10_Abundance"],
        s=42,
        alpha=0.55,
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel(xlabels[comm])
    ax_main.set_ylabel("")
    ax_main.tick_params(axis="y", left=False, labelleft=False)
    style_ax(ax_main, grid=False)

    ax_theory = ax_main.twinx()
    ax_theory.set_zorder(ax_main.get_zorder() + 1)
    ax_theory.patch.set_alpha(0.0)
    ax_theory.set_ylabel("Predicted abundance", color=theory_line_color, labelpad=12)
    ax_theory.tick_params(axis="y", colors=theory_line_color, width=0.4, pad=6)
    ax_theory.grid(False)

    for spine in ax_theory.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.4)

    x_grid = np.linspace(dat_full["Species_CUE"].min(), dat_full["Species_CUE"].max(), 800)
    delta = np.maximum(x_grid - eps_c, 0.0)
    c_theory = Cmax * (1.0 - np.exp(-delta / H))
    valid = c_theory > SURVIVAL_THRESHOLD

    ax_theory.set_yscale("log")
    if valid.sum() > 0:
        ax_theory.set_ylim(np.min(c_theory[valid]), np.max(c_theory[valid]) * 1.05)
        ax_theory.plot(
            x_grid[valid],
            c_theory[valid],
            color=theory_line_color,
            linewidth=2.8,
            zorder=10
        )
    else:
        ax_theory.set_ylim(SURVIVAL_THRESHOLD, 1.0)

plt.tight_layout()
plt.show()




# ====================================================================================================
# ========================= Competition vs Community-level CUE =================================
# ====================================================================================================


from matplotlib.ticker import MaxNLocator

df_comm_agg = (
    df_surv
    .groupby(["Seed", "Community", "Competition", "Community_CUE_surv", "Facilitation"], as_index=False)
    .agg(Species_CUE_Var=("Species_CUE", lambda x: np.nanvar(x, ddof=1)))
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

for i, (ax, comm) in enumerate(zip(axes, ["1", "2", "3"])):
    dat = df_comm_agg[df_comm_agg["Community"] == comm]

    ax.scatter(
        dat["Competition"],
        dat["Community_CUE_surv"],
        s=44,
        alpha=0.6,
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    ax.set_xlabel("Community uptake similarity")
    ax.set_title(community_labels[comm], pad=8)
    ax.set_ylim(0.53, 0.565)

    # 每个 x 轴只保留 4 个主刻度
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # 只有最左侧显示 y 轴刻度和标签
    if i == 0:
        ax.set_ylabel("Community-level CUE")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    style_ax(ax, grid=False)

plt.tight_layout()
plt.show()



# ====================================================================================================
# =========================== Species competition vs Species CUE ================================
# ====================================================================================================

from matplotlib.ticker import MaxNLocator, ScalarFormatter

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

for i, (ax, comm) in enumerate(zip(axes, ["1", "2", "3"])):
    dat = df_surv[df_surv["Community"] == comm]

    ax.scatter(
        dat["Species_Competition2"],
        dat["Species_CUE"],
        s=40,
        alpha=0.55,
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    ax.set_xlabel("Species uptake similarity")
    ax.set_title(community_labels[comm], pad=8)
    ax.set_ylim(0.50, 0.59)

    # x轴只保留4个主刻度
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # x轴使用科学计数法，并在轴旁边显示类似 1e-3 的标记
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)

    # 调整偏移文字大小，也就是右下角那个 1e-3
    ax.xaxis.get_offset_text().set_fontsize(12)

    if i == 0:
        ax.set_ylabel("Species-level CUE")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    style_ax(ax, grid=False)

plt.tight_layout()
plt.show()




# ====================================================================================================
# ========================== Facilitation vs Community-level CUE =================================
# ====================================================================================================

from matplotlib.ticker import MaxNLocator, ScalarFormatter

df_comm_fac = (
    df_surv
    .groupby(["Seed", "Community"], as_index=False)
    .agg(
        Community_CUE_surv=("Community_CUE_surv", first_unique),
        Facilitation=("Facilitation", "mean")
    )
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)

for i, (ax, comm) in enumerate(zip(axes, ["1", "2", "3"])):
    dat = df_comm_fac[df_comm_fac["Community"] == comm]

    ax.scatter(
        dat["Facilitation"],
        dat["Community_CUE_surv"],
        s=44,
        alpha=0.6,
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        zorder=3
    )

    ax.set_xlabel("Facilitation")
    ax.set_title(community_labels[comm], pad=8)
    ax.set_ylim(0.53, 0.565)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(12)

    if i == 0:
        ax.set_ylabel("Community-level CUE")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    style_ax(ax, grid=False)

plt.tight_layout()
plt.show()




# ====================================================================================================
# ====================== ΔCUE vs ΔSimilarity with Dominance ====================================
# ====================================================================================================

df_mut = df_surv.copy()
df_mut["Global_Species_ID"] = np.where(
    df_mut["Community"] == "2",
    df_mut["Species_ID"] + 100,
    df_mut["Species_ID"]
)

bray_rows = []

for s in sorted(df_mut["Seed"].unique()):
    df_seed = df_mut[df_mut["Seed"] == s].copy()

    comm_mat = (
        df_seed[["Community", "Global_Species_ID", "Abundance"]]
        .pivot_table(
            index="Community",
            columns="Global_Species_ID",
            values="Abundance",
            aggfunc="sum",
            fill_value=0
        )
        .reindex(["1", "2", "3"])
        .fillna(0)
    )

    bc = squareform(pdist(comm_mat.values, metric="braycurtis"))

    d31 = bc[2, 0]
    d32 = bc[2, 1]

    cue1 = first_unique(df_seed.loc[df_seed["Community"] == "1", "Community_CUE_surv"])
    cue2 = first_unique(df_seed.loc[df_seed["Community"] == "2", "Community_CUE_surv"])

    bray_rows.append({
        "Seed": s,
        "Bray_3vs1": d31,
        "Bray_3vs2": d32,
        "CUE_1": cue1,
        "CUE_2": cue2,
        "Sim_3vs1": 1 - d31,
        "Sim_3vs2": 1 - d32
    })

bray_results = pd.DataFrame(bray_rows)

df_comm = (
    df[df["Community"].isin(["1", "2"])]
    .groupby(["Seed", "Community"], as_index=False)
    .agg(
        Community_CUE_surv=("Community_CUE_surv", first_unique),
        Dominant_Community=("Dominant_Community", first_unique)
    )
)

df_diff = bray_results.copy()
df_diff["CUE_Diff"] = df_diff["CUE_1"] - df_diff["CUE_2"]
df_diff["Sim_Diff"] = df_diff["Sim_3vs1"] - df_diff["Sim_3vs2"]

dom_seed = (
    df_comm[df_comm["Community"] == "1"][["Seed", "Dominant_Community"]]
    .drop_duplicates()
    .copy()
)

df_diff = df_diff.merge(dom_seed, on="Seed", how="left")
df_diff["DomGroup"] = np.select(
    [
        df_diff["Dominant_Community"] == "Community 1",
        df_diff["Dominant_Community"] == "Community 2"
    ],
    [
        "Community 1",
        "Community 2"
    ],
    default="Neither"
)

dom_colors = {
    "Community 1": pal_rgb["1"],
    "Community 2": pal_rgb["2"]
}

fig, ax = plt.subplots(figsize=(8, 5))

for grp in ["Community 1", "Community 2"]:
    dat = df_diff[df_diff["DomGroup"] == grp]

    ax.scatter(
        dat["CUE_Diff"],
        dat["Sim_Diff"],
        s=100,
        alpha=0.65,
        facecolors=dom_colors[grp],
        edgecolors="black",
        linewidths=0.5,
        label=grp,
        zorder=3
    )

ax.axhline(0, linestyle="--", color="black", linewidth=0.7)
ax.set_xlim(-0.02, 0.02)
ax.set_xlabel(r"Community CUE difference")
ax.set_ylabel(r"Similarity difference")
ax.legend(
    title="Dominant community",
    loc="lower right",
    frameon=True,
    edgecolor="black",
    framealpha=0.6
)
style_ax(ax, grid=False)

plt.tight_layout()
plt.show()



# ====================================================================================================
# ==================== ΔCUE vs ΔSimilarity under Resource Overlap ===============================
# ====================================================================================================

df_diff_resource = df_resource.copy()
df_diff_resource["CUE_diff"] = df_diff_resource["CUE1"] - df_diff_resource["CUE2"]
df_diff_resource["Sim_diff"] = df_diff_resource["Sim_3vs1"] - df_diff_resource["Sim_3vs2"]

df_diff_resource = df_diff_resource[
    (df_diff_resource["CUE_diff"] >= -0.02) & 
    (df_diff_resource["CUE_diff"] <= 0.02)
]

n_bins = 7
#breaks = np.linspace(df_diff_resource["CUE_diff"].min(), df_diff_resource["CUE_diff"].max(), n_bins + 1)
breaks = np.linspace(-0.02, 0.02, n_bins + 1)

df_diff_resource["CUE_bin"] = pd.cut(
    df_diff_resource["CUE_diff"],
    bins=breaks,
    include_lowest=True
)

interval_order = sorted(
    df_diff_resource["CUE_bin"].dropna().unique(),
    key=lambda x: x.mid
)

df_diff_resource["CUE_bin"] = pd.Categorical(
    df_diff_resource["CUE_bin"],
    categories=interval_order,
    ordered=True
)

overlap_colors = {
    "0.25": "#D8A39A",
    "0.5": "#D9C27A",
    "0.75": "#9FB7CC"
}

fig, ax = plt.subplots(figsize=(10.2, 5.2))

sns.boxplot(
    data=df_diff_resource,
    x="CUE_bin",
    y="Sim_diff",
    hue="Overlap",
    palette=overlap_colors,
    dodge=True,
    width=0.72,
    fliersize=1.8,
    linewidth=0.7,
    saturation=1,
    ax=ax,
    boxprops=dict(edgecolor="black"),
    medianprops=dict(color="black", linewidth=0.9),
    whiskerprops=dict(color="black", linewidth=0.7),
    capprops=dict(color="black", linewidth=0.7),
    flierprops=dict(
        marker="o",
        markersize=2.2,
        markerfacecolor="white",
        markeredgecolor="black",
        alpha=0.8
    )
)

ax.axhline(0, linestyle="--", color="black", linewidth=0.7)
ax.set_ylim(-0.4, 0.4)
ax.set_xlabel(r"Community CUE difference")
ax.set_ylabel(r"Similarity difference")
lg = ax.legend(
    title="Resource overlap",
    loc="upper left",
    frameon=True,
    edgecolor="black",
    facecolor="white",
    framealpha=0.6
)
lg.get_texts()[0].set_text("25%")
lg.get_texts()[1].set_text("50%")
lg.get_texts()[2].set_text("75%")

ax.set_xticklabels(
    [f"({iv.left:.2f}, {iv.right:.2f}]" for iv in interval_order],
    rotation=40,
    ha="right"
)
style_ax(ax, grid=False)

plt.tight_layout()
plt.show()




# ====================================================================================================
# ============================= Community-level CUE vs Depletion ================================
# ====================================================================================================

df_depletion = (
    df.groupby(["Seed", "Community"], as_index=False)
    .agg(
        Community_CUE_surv=("Community_CUE_surv", first_unique),
        Niche_Overlap=("Competition", first_unique),
        Depletion=("Depletion", first_unique)
    )
)

marker_map = {"1": "o", "2": "^", "3": "s"}

fig, axes = plt.subplots(
    1, 2,
    figsize=(10.2, 4.8),
    gridspec_kw={"width_ratios": [2, 1]}
)

ax1, ax2 = axes

for comm in ["1", "2", "3"]:
    dat = df_depletion[df_depletion["Community"] == comm]

    ax1.scatter(
        dat["Community_CUE_surv"],
        dat["Depletion"],
        s=48,
        alpha=0.65,
        marker=marker_map[comm],
        facecolors=pal_rgb[comm],
        edgecolors="black",
        linewidths=0.5,
        label=community_labels[comm],
        zorder=3
    )

ax1.set_xlabel("Community-level CUE")
ax1.set_ylabel("Total residual resource")
ax1.legend(
    loc="upper right",
    frameon=True,
    edgecolor="black",
    facecolor="white",
    framealpha=0.6
)
style_ax(ax1, grid=False)

sns.boxplot(
    data=df_depletion,
    x="Community",
    y="Community_CUE_surv",
    palette=pal_rgb,
    width=0.62,
    fliersize=1.8,
    linewidth=0.7,
    saturation=1,
    ax=ax2,
    boxprops=dict(edgecolor="black"),
    medianprops=dict(color="black", linewidth=0.9),
    whiskerprops=dict(color="black", linewidth=0.7),
    capprops=dict(color="black", linewidth=0.7),
    flierprops=dict(
        marker="o",
        markersize=2.2,
        markerfacecolor="white",
        markeredgecolor="black",
        alpha=0.8
    )
)

ax2.set_xticklabels(["Community 1", "Community 2", "Coalescence"], rotation=15)
ax2.set_xlabel("")
ax2.set_ylabel("Community-level CUE")
style_ax(ax2, grid=False)

plt.tight_layout()
plt.show()




# ====================================================================================================
# ============================== Rare Species Invasion ==========================================
# ====================================================================================================

df_rare["survival"] = np.where(df_rare["C_final"] > SURVIVAL_THRESHOLD, "Survived", "Extinct")
df_rare_filt = df_rare[df_rare["DilutionRate"].isin([0.01, 0.1])].copy()

n_bins = 20
df_rare_filt["CUE_bin"] = pd.cut(df_rare_filt["CUE"], bins=n_bins)

df_rare_bar = (
    df_rare_filt
    .groupby(["DilutionRate", "CUE_bin", "survival"], observed=False)
    .size()
    .reset_index(name="count")
)

interval_order = sorted(
    df_rare_bar["CUE_bin"].dropna().unique(),
    key=lambda x: x.mid
)

dilution_labels = {
    0.01: "Rarity level = 0.01",
    0.1: "Rarity level = 0.1"
}

color_map = {
    "Survived": "#A8C3A6",
    "Extinct": "#D8A39A"
}

# 断轴范围，可根据你的数据再微调
y_low_max = 550
y_high_min = 2200
y_high_max = 2600

fig = plt.figure(figsize=(12, 5.8))
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1, 4],
    wspace=0.08,
    hspace=0.03
)

# 上排
ax_top_left = fig.add_subplot(gs[0, 0])
ax_top_right = fig.add_subplot(gs[0, 1], sharey=ax_top_left)

# 下排
ax_bot_left = fig.add_subplot(gs[1, 0], sharex=ax_top_left)
ax_bot_right = fig.add_subplot(gs[1, 1], sharex=ax_top_right, sharey=ax_bot_left)

top_axes = [ax_top_left, ax_top_right]
bot_axes = [ax_bot_left, ax_bot_right]

for i, dil in enumerate([0.01, 0.1]):
    ax_top = top_axes[i]
    ax_bot = bot_axes[i]

    dat = df_rare_bar[df_rare_bar["DilutionRate"] == dil].copy()

    pivot = (
        dat.pivot_table(
            index="CUE_bin",
            columns="survival",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        .reindex(interval_order)
        .fillna(0)
    )

    counts = pivot.reindex(columns=["Extinct", "Survived"], fill_value=0)
    extinct = counts["Extinct"].to_numpy()
    survived = counts["Survived"].to_numpy()

    surv_mask = survived > 0
    first_surv_idx = int(np.argmax(np.r_[surv_mask, True]))

    plot_extinct_full = np.concatenate(([extinct[:first_surv_idx].sum()], extinct[first_surv_idx:]))
    plot_survived_full = np.concatenate(([0.0], survived[first_surv_idx:]))

    drop_leading = int(first_surv_idx == 0)
    plot_extinct = plot_extinct_full[drop_leading:]
    plot_survived = plot_survived_full[drop_leading:]

    merged_label = f"< {interval_order[first_surv_idx].left:.2f}"
    tick_labels_full = [merged_label] + [
        f"{iv.left:.2f}" for iv in interval_order[first_surv_idx:]
    ]
    tick_labels = tick_labels_full[drop_leading:]

    x = np.arange(len(plot_extinct))

    # 下半部分
    ax_bot.bar(
        x, plot_extinct,
        color=color_map["Extinct"],
        edgecolor="black",
        linewidth=0.35,
        label="Extinct",
        width=0.85
    )
    ax_bot.bar(
        x, plot_survived,
        bottom=plot_extinct,
        color=color_map["Survived"],
        edgecolor="black",
        linewidth=0.35,
        label="Survived",
        width=0.85
    )

    # 上半部分
    ax_top.bar(
        x, plot_extinct,
        color=color_map["Extinct"],
        edgecolor="black",
        linewidth=0.35,
        width=0.85
    )
    ax_top.bar(
        x, plot_survived,
        bottom=plot_extinct,
        color=color_map["Survived"],
        edgecolor="black",
        linewidth=0.35,
        width=0.85
    )

    ax_top.set_title(dilution_labels[dil], pad=8)

    ax_bot.set_xlabel("Species-level CUE")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(tick_labels, rotation=40, ha="right")

    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_bot.set_ylim(0, y_low_max)
    ax_top.set_ylim(y_high_min, y_high_max)

    style_ax(ax_top, grid=False)
    style_ax(ax_bot, grid=False)

    # 关键：隐藏中间那两条横线
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)

    # 左图保留 y 轴，右图隐藏 y 轴
    if i == 0:
        ax_bot.set_ylabel("Number of species")
    else:
        ax_bot.set_ylabel("")
        ax_bot.tick_params(axis="y", left=False, labelleft=False)
        ax_top.tick_params(axis="y", left=False, labelleft=False)

    # 每个子图都放图例，加框线
    handles, labels = ax_bot.get_legend_handles_labels()
    leg = ax_bot.legend(
        handles[::-1], labels[::-1],
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.6
    )
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

# 只在左侧 y 轴画断裂斜杠
d = 0.012
kwargs_top = dict(transform=ax_top_left.transAxes, color="black", clip_on=False, linewidth=1.2)
kwargs_bot = dict(transform=ax_bot_left.transAxes, color="black", clip_on=False, linewidth=1.2)

# 左图上轴左侧斜杠
ax_top_left.plot((-d, +d), (-d, +d), **kwargs_top)

# 左图下轴左侧斜杠
ax_bot_left.plot((-d, +d), (1 - d, 1 + d), **kwargs_bot)

plt.tight_layout()
plt.show()