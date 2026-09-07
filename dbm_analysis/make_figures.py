"""Flat DBM plotting script; run sections interactively or as a script."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from dbm_results import load_best_dbm_fits


# Configuration ---------------------------------------------------------------
analysis_dir = Path(__file__).resolve().parent
base_dir = analysis_dir.parent
figure_dir = base_dir / "figures"
figure_dir.mkdir(exist_ok=True)
family_order = ["guessing", "rule-based", "procedural"]
family_labels = {"guessing": "Guessing", "rule-based": "Rule-based", "procedural": "Procedural"}
family_colours = {"guessing": "#7F7F7F", "rule-based": "#377EB8", "procedural": "#E41A1C"}
condition_order = ["90", "180"]
# Each lab day is followed by three 400-trial at-home days.
# Planned design: eight lab days, each followed by three 400-trial home days.
lab_days = {session: 1 + 4 * (session - 1) for session in range(1, 9)}
home_days = {
    session: session + ((session - 1) // 3) + 1
    for session in range(1, 25)
}
# Explicit probe marker audit: the 50 normal training trials immediately
# before the probe correspond to these phase-order training blocks.
pre_probe_train_block = {1: "train_12", 2: "train_03", 3: "train_06", 4: "train_08", 5: "train_04"}


# Load best fits --------------------------------------------------------------
lab_best = load_best_dbm_fits(context="lab")
home_best = load_best_dbm_fits(context="home")
lab_best["probe_condition"] = lab_best["probe_condition"].astype(str)
home_best["probe_condition"] = home_best["probe_condition"].astype(str)


# Figure: condition-split DBM family trajectories across training ------------
# Each point is the proportion of 50-trial blocks won by each DBM family.
lab_train = lab_best.loc[lab_best["phase"].eq("train")].copy()
lab_train["timeline_day"] = lab_train["session"].map(lab_days)
home_train = home_best.copy()
home_train["timeline_day"] = home_train["session"].map(home_days)
training = pd.concat([lab_train, home_train], ignore_index=True)
training["block_number"] = training["block"].str.extract(r"(\d+)$").astype(int)
training["blocks_that_day"] = np.where(training["context"].eq("lab"), 12, 8)
training["experiment_block"] = training.apply(
    lambda row: sum(12 if day in lab_days.values() else 8 for day in range(1, int(row.timeline_day)))
    + row.block_number,
    axis=1,
)
active_days = list(range(1, int(training["timeline_day"].max()) + 1))
active_lab_days = [day for day in lab_days.values() if day in active_days]
blocks_per_day = {day: 12 if day in lab_days.values() else 8 for day in active_days}
day_start_block = {}
block_offset = 0
for day in active_days:
    day_start_block[day] = block_offset
    block_offset += blocks_per_day[day]
total_experiment_blocks = block_offset

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, condition in zip(axes, condition_order):
    subset = training.loc[training["probe_condition"].eq(condition)]
    counts = subset.groupby(["experiment_block", "best_model_class"]).size().rename("n").reset_index()
    totals = subset.groupby("experiment_block").size().rename("total").reset_index()
    counts = counts.merge(totals, on="experiment_block")
    counts["proportion"] = counts["n"] / counts["total"]
    for family in family_order:
        line = counts.loc[counts["best_model_class"].eq(family)].set_index("experiment_block")["proportion"]
        ax.plot(line.index, line.values, marker="o", linewidth=1.5,
                color=family_colours[family], label=family_labels[family])
    for day in sorted(set(lab_train["timeline_day"].dropna())):
        boundary = sum(12 if d in lab_days.values() else 8 for d in range(1, day)) + 0.5
        ax.axvline(boundary, color="black", linewidth=0.8, alpha=0.5)
    ax.set(title=f"Probe condition {condition}°", xlabel="Chronological 50-trial training block",
           ylim=(0, 1), ylabel="Proportion of blocks")
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, title="Winning family")
fig.suptitle("DBM family winners across training, split by probe condition")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_family_training_by_probe_condition.png", dpi=300)
plt.close(fig)


# Figure: overall family proportions by chronological block ------------------
block_counts = (training.groupby(["experiment_block", "best_model_class"]).size()
                .rename("n").reset_index())
block_totals = training.groupby("experiment_block").size().rename("total").reset_index()
block_counts = block_counts.merge(block_totals, on="experiment_block")
block_counts["proportion"] = block_counts["n"] / block_counts["total"]
fig, ax = plt.subplots(figsize=(15, 4.8))
for day in active_lab_days:
    ax.axvspan(day_start_block[day] + 0.5, day_start_block[day] + blocks_per_day[day] + 0.5,
               color="#F2F2F2", zorder=0)
for family in family_order:
    values = (block_counts.loc[block_counts["best_model_class"].eq(family)]
              .set_index("experiment_block")["proportion"]
              .reindex(range(1, total_experiment_blocks + 1), fill_value=0))
    ax.plot(values.index, values, linewidth=1.4, color=family_colours[family], label=family_labels[family])
ax.set(xlabel="Chronological 50-trial block (grey = lab day)", ylabel="Proportion of participants",
       ylim=(0, 1), xlim=(0.5, total_experiment_blocks + 0.5))
ax.set_xticks([day_start_block[day] + (blocks_per_day[day] + 1) / 2 for day in active_days], active_days)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, title="Winning family")
ax.set_title("perc DBM family winners by chronological 50-trial block")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_family_block_proportions.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure: family proportions by experiment day -------------------------------
day_counts = (training.groupby(["timeline_day", "best_model_class"]).size()
              .rename("n").reset_index())
day_totals = training.groupby("timeline_day").size().rename("total").reset_index()
day_counts = day_counts.merge(day_totals, on="timeline_day")
day_counts["proportion"] = day_counts["n"] / day_counts["total"]
fig, ax = plt.subplots(figsize=(12, 4.8))
for day in active_lab_days:
    ax.axvspan(day - 0.5, day + 0.5, color="#F2F2F2", zorder=0)
for family in family_order:
    values = (day_counts.loc[day_counts["best_model_class"].eq(family)]
              .set_index("timeline_day")["proportion"].reindex(active_days, fill_value=0))
    ax.plot(active_days, values, marker="o", linewidth=2, color=family_colours[family], label=family_labels[family])
ax.set(xlabel="Experiment day (grey = lab day)", ylabel="Proportion of 50-trial blocks",
       ylim=(0, 1), xlim=(0.5, max(active_days) + 0.5))
ax.set_xticks(active_days)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, title="Winning family")
ax.set_title("perc DBM family winners by experiment day")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_family_day_proportions.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure: lab-only family proportions ----------------------------------------
lab_day_counts = (lab_train.groupby(["timeline_day", "best_model_class"]).size()
                  .rename("n").reset_index())
lab_day_totals = lab_train.groupby("timeline_day").size().rename("total").reset_index()
lab_day_counts = lab_day_counts.merge(lab_day_totals, on="timeline_day")
lab_day_counts["proportion"] = lab_day_counts["n"] / lab_day_counts["total"]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for family in family_order:
    values = (lab_day_counts.loc[lab_day_counts["best_model_class"].eq(family)]
              .set_index("timeline_day")["proportion"].reindex(active_lab_days, fill_value=0))
    ax.plot(active_lab_days, values, marker="o", linewidth=2.4, color=family_colours[family], label=family_labels[family])
ax.set(xlabel="Experiment day (lab sessions)", ylabel="Proportion of 50-trial blocks",
       ylim=(0, 1), xlim=(0.5, max(active_days) + 0.5))
ax.set_xticks(active_lab_days)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, title="Winning family")
ax.set_title("perc lab DBM family winners by experiment day")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_lab_family_day_proportions.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure: participant-day dominant family ------------------------------------
subjects = sorted(training["subject"].unique())
family_code = {"guessing": 0, "rule-based": 1, "procedural": 2}
dominant_grid = np.full((len(subjects), len(active_days)), np.nan)
tie_hatches = {}
for row_index, subject in enumerate(subjects):
    for column_index, day in enumerate(active_days):
        families = training.loc[
            training["subject"].eq(subject) & training["timeline_day"].eq(day),
            "best_model_class",
        ].astype(str)
        if families.empty:
            continue
        family_counts = families.value_counts()
        winners = sorted(family_counts.loc[family_counts.eq(family_counts.max())].index.tolist())
        if len(winners) == 1:
            dominant_grid[row_index, column_index] = family_code[winners[0]]
        else:
            dominant_grid[row_index, column_index] = 3
            tie_hatches[(row_index, column_index)] = {
                frozenset(["procedural", "rule-based"]): "/",
                frozenset(["procedural", "guessing"]): "\\\\",
                frozenset(["rule-based", "guessing"]): "x",
                frozenset(["procedural", "rule-based", "guessing"]): "+",
            }.get(frozenset(winners), "o")
colour_map = ListedColormap([family_colours["guessing"], family_colours["rule-based"],
                              family_colours["procedural"], "#9C6ADE"])
colour_map.set_bad("white")
fig, ax = plt.subplots(figsize=(max(12, len(active_days) * 0.7), max(5, len(subjects) * 0.45)))
ax.imshow(dominant_grid, aspect="auto", interpolation="none", cmap=colour_map, vmin=0, vmax=3)
for day in active_lab_days:
    column_index = active_days.index(day)
    ax.add_patch(Rectangle((column_index - 0.5, -0.5), 1, len(subjects), fill=False,
                           edgecolor="black", linewidth=1.2))
for (row_index, column_index), hatch in tie_hatches.items():
    ax.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, fill=False,
                           hatch=hatch, edgecolor="black", linewidth=0.5))
ax.set(xlabel="Experiment day (black outlines = lab day)", ylabel="Participant")
ax.set_xticks(range(len(active_days)), active_days)
ax.set_yticks(range(len(subjects)), subjects)
ax.legend(handles=[
    Patch(facecolor=family_colours["guessing"], label="Guessing"),
    Patch(facecolor=family_colours["rule-based"], label="Rule-based"),
    Patch(facecolor=family_colours["procedural"], label="Procedural"),
    Patch(facecolor="#9C6ADE", hatch="/", label="Family tie (hatch specifies tie)"),
], frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.12))
ax.set_title("perc participant-day dominant DBM family")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_family_by_participant.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure: lab-day participant dominant-family counts -------------------------
lab_dominant = []
for subject in subjects:
    for day in active_lab_days:
        families = lab_train.loc[
            lab_train["subject"].eq(subject) & lab_train["timeline_day"].eq(day),
            "best_model_class",
        ].astype(str)
        if families.empty:
            continue
        counts = families.value_counts()
        winners = sorted(counts.loc[counts.eq(counts.max())].index.tolist())
        lab_dominant.append({"timeline_day": day, "family": winners[0] if len(winners) == 1 else "Family tie"})
lab_table = (pd.DataFrame(lab_dominant).groupby(["timeline_day", "family"]).size()
             .unstack(fill_value=0).reindex(index=active_lab_days,
             columns=["guessing", "rule-based", "procedural", "Family tie"], fill_value=0))
lab_table.columns = ["Guessing", "Rule-based", "Procedural", "Family tie"]
lab_table["Total"] = lab_table.sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")
table = ax.table(cellText=lab_table.to_numpy(), rowLabels=[f"Day {day}" for day in lab_table.index],
                 colLabels=lab_table.columns, cellLoc="center", rowLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.55)
ax.set_title("perc lab days: participants' dominant DBM family", pad=14)
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_lab_participant_family_counts_table.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure: immediately-pre-probe training to probe strategy transitions -------
# Each 50-trial probe is compared with the 50 normal training trials directly
# before it in original trial order.  The mapping above follows the explicit
# perc probe markers, rather than a fixed terminal-trial assumption.
probe_pair = lab_best.loc[lab_best["block"].eq("probe_01")].copy()
train_pair = lab_best.loc[lab_best["phase"].eq("train")].copy()
train_pair["pre_probe_block"] = train_pair["session"].map(pre_probe_train_block)
train_pair = train_pair.loc[train_pair["block"].eq(train_pair["pre_probe_block"])]
pair_keys = ["subject", "session", "probe_condition"]
train_class = (train_pair.groupby(pair_keys)["best_model_class"].agg(
    lambda x: sorted(x.value_counts().loc[lambda n: n.eq(n.max())].index.astype(str))[0]
).rename("train_family").reset_index())
probe_class = (probe_pair.groupby(pair_keys)["best_model_class"].agg(
    lambda x: sorted(x.value_counts().loc[lambda n: n.eq(n.max())].index.astype(str))[0]
).rename("probe_family").reset_index())
transitions = train_class.merge(probe_class, on=pair_keys, how="inner")

lab_sessions = sorted(lab_best["session"].unique())
fig, axes = plt.subplots(2, len(lab_sessions), figsize=(3.6 * len(lab_sessions), 7),
                         sharex=True, sharey=True, squeeze=False)
heatmap_colours = ListedColormap(["#F7FBFF", "#9ECAE1", "#3182BD", "#08519C"])
for column, session in enumerate(lab_sessions):
    for row, condition in enumerate(condition_order):
        ax = axes[row, column]
        subset = transitions.loc[
            transitions["session"].eq(session) & transitions["probe_condition"].eq(condition)
        ]
        counts = pd.crosstab(subset["probe_family"], subset["train_family"]).reindex(
            index=family_order, columns=family_order, fill_value=0
        )
        ax.imshow(counts.values, cmap=heatmap_colours, vmin=0, vmax=max(1, counts.values.max()))
        for y in range(3):
            for x in range(3):
                ax.text(x, y, str(counts.iat[y, x]), ha="center", va="center", fontsize=10)
        ax.set_xticks(range(3), ["GS", "RB", "PR"])
        ax.set_yticks(range(3), ["GS", "RB", "PR"])
        if row == 0:
            ax.set_title(f"Lab day {lab_days[session]}")
        if column == 0:
            ax.set_ylabel(f"{condition}°\nProbe")
        if row == 1:
            ax.set_xlabel("Immediately preceding training")
fig.suptitle("DBM family transitions: preceding 50 training trials to 50-trial probe")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_probe_transition_heatmaps_by_day_condition.png", dpi=300)
plt.close(fig)


# Figure: DBM recovery heatmaps ----------------------------------------------
lab_family_props = pd.read_csv(
    analysis_dir / "dbm_fits/recovery_lab/fit_dbm_recovery_block_all_family_props.csv",
    index_col=0,
)
home_family_props = pd.read_csv(
    analysis_dir / "dbm_fits/recovery_home/fit_dbm_recovery_block_all_family_props.csv",
    index_col=0,
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True, sharey=True)
for ax, recovery_context, family_props in zip(
    axes, ["Lab", "At-home"], [lab_family_props, home_family_props]
):
    ax.imshow(family_props.values, cmap="Blues", vmin=0, vmax=1)
    for y in range(family_props.shape[0]):
        for x in range(family_props.shape[1]):
            ax.text(x, y, f"{family_props.iat[y, x]:.2f}", ha="center", va="center")
    ax.set_xticks(range(family_props.shape[1]), family_props.columns, rotation=30, ha="right")
    ax.set_yticks(range(family_props.shape[0]), family_props.index)
    ax.set(xlabel="Recovered family", ylabel="True family", title=f"{recovery_context} recovery")
fig.suptitle("perc DBM recovery: family confusion")
fig.tight_layout()
fig.savefig(figure_dir / "perc_dbm_recovery_family_props.png", dpi=300)
plt.close(fig)
