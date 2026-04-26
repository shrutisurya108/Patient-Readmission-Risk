# src/ingestion/eda.py

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml

from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DtypeWarning.*")

logger = get_logger("eda")

# ── Consistent visual style ──────────────────────────────────────────────────
PALETTE   = {"NO": "#4CAF50", ">30": "#FF9800", "<30": "#F44336"}
FIG_DPI   = 150
TITLE_FS  = 14
LABEL_FS  = 11
sns.set_theme(style="whitegrid", font_scale=1.05)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ── 1. Class Distribution ────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, out_dir: str) -> None:
    counts = df["readmitted"].value_counts().reindex(["NO", ">30", "<30"])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 400,
                f"{val:,}\n({val/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("Readmission Class Distribution", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Readmission Status", fontsize=LABEL_FS)
    ax.set_ylabel("Patient Count", fontsize=LABEL_FS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, os.path.join(out_dir, "eda_class_distribution.png"))


# ── 2. Missing Values Heatmap ────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame, out_dir: str) -> None:
    # Replace "?" with NaN for missingness analysis
    df_miss = df.replace("?", np.nan)
    miss_pct = (df_miss.isnull().sum() / len(df_miss) * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]

    if miss_pct.empty:
        logger.info("No missing values detected — skipping missing values plot.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(miss_pct) * 0.4)))
    colors = ["#F44336" if v > 40 else "#FF9800" if v > 10 else "#4CAF50"
              for v in miss_pct.values]
    bars = ax.barh(miss_pct.index, miss_pct.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, miss_pct.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.axvline(40, color="#F44336", linestyle="--", linewidth=1.2, label="40% threshold (drop)")
    ax.set_title("Missing Values by Feature (% of rows)", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Missing %", fontsize=LABEL_FS)
    ax.legend(fontsize=9)
    _save(fig, os.path.join(out_dir, "eda_missing_values.png"))


# ── 3. Age Distribution by Readmission Class ─────────────────────────────────
def plot_age_distribution(df: pd.DataFrame, out_dir: str) -> None:
    age_order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    plot_df = (df.groupby(["age", "readmitted"])
                 .size()
                 .reset_index(name="count"))
    plot_df["age"] = pd.Categorical(plot_df["age"], categories=age_order, ordered=True)
    plot_df = plot_df.sort_values("age")

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(age_order))
    for status, color in PALETTE.items():
        vals = []
        for age in age_order:
            sub = plot_df[(plot_df["age"] == age) & (plot_df["readmitted"] == status)]
            vals.append(sub["count"].values[0] if not sub.empty else 0)
        ax.bar(age_order, vals, bottom=bottom, label=status, color=color, edgecolor="white")
        bottom += np.array(vals)

    ax.set_title("Patient Age Distribution by Readmission Status", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Age Group", fontsize=LABEL_FS)
    ax.set_ylabel("Patient Count", fontsize=LABEL_FS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Readmitted", fontsize=LABEL_FS)
    plt.xticks(rotation=30, ha="right")
    _save(fig, os.path.join(out_dir, "eda_age_distribution.png"))


# ── 4. Numeric Feature Distributions ────────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame, out_dir: str) -> None:
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses"
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        for status, color in PALETTE.items():
            subset = df[df["readmitted"] == status][col].dropna()
            ax.hist(subset, bins=25, alpha=0.55, color=color,
                    label=status, edgecolor="none", density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, title="Readmitted")

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Readmission Status",
                 fontsize=TITLE_FS, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "eda_numeric_distributions.png"))


# ── 5. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix (Numeric Features)",
                 fontsize=TITLE_FS, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    _save(fig, os.path.join(out_dir, "eda_correlation_heatmap.png"))


# ── 6. Readmission Rate vs Time in Hospital ──────────────────────────────────
def plot_readmit_by_time(df: pd.DataFrame, out_dir: str) -> None:
    grp = (df.groupby("time_in_hospital")
             .apply(lambda x: (x["readmitted"] == "<30").sum() / len(x) * 100,
                    include_groups=False)
             .reset_index(name="readmit_rate_pct"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grp["time_in_hospital"], grp["readmit_rate_pct"],
            marker="o", color="#F44336", linewidth=2.2, markersize=7)
    ax.fill_between(grp["time_in_hospital"], grp["readmit_rate_pct"],
                    alpha=0.12, color="#F44336")
    ax.set_title("30-Day Readmission Rate vs Time in Hospital",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Time in Hospital (days)", fontsize=LABEL_FS)
    ax.set_ylabel("Readmission Rate (%)", fontsize=LABEL_FS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(True, alpha=0.4)
    _save(fig, os.path.join(out_dir, "eda_readmit_by_time.png"))


# ── 7. Readmission Rate vs Prior Inpatient Visits ────────────────────────────
def plot_readmit_by_inpatient(df: pd.DataFrame, out_dir: str) -> None:
    grp = (df[df["number_inpatient"] <= 10]
             .groupby("number_inpatient")
             .apply(lambda x: (x["readmitted"] == "<30").sum() / len(x) * 100,
                    include_groups=False)
             .reset_index(name="readmit_rate_pct"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grp["number_inpatient"], grp["readmit_rate_pct"],
           color="#FF9800", edgecolor="white", width=0.6)
    for _, row in grp.iterrows():
        ax.text(row["number_inpatient"], row["readmit_rate_pct"] + 0.4,
                f"{row['readmit_rate_pct']:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("30-Day Readmission Rate by Prior Inpatient Visits",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Number of Prior Inpatient Visits", fontsize=LABEL_FS)
    ax.set_ylabel("Readmission Rate (%)", fontsize=LABEL_FS)
    ax.set_xticks(grp["number_inpatient"])
    _save(fig, os.path.join(out_dir, "eda_readmit_by_inpatient.png"))


# ── 8. Top Medications by Readmission Rate ───────────────────────────────────
def plot_top_medications(df: pd.DataFrame, out_dir: str) -> None:
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "glipizide", "glyburide", "pioglitazone",
        "rosiglitazone", "acarbose", "insulin", "tolbutamide",
        "miglitol", "troglitazone", "tolazamide", "glyburide-metformin"
    ]
    med_cols = [c for c in med_cols if c in df.columns]

    rates = {}
    for med in med_cols:
        changed = df[df[med].isin(["Up", "Down"])]
        if len(changed) > 50:
            rate = (changed["readmitted"] == "<30").sum() / len(changed) * 100
            rates[med] = rate

    rates_df = (pd.DataFrame.from_dict(rates, orient="index", columns=["readmit_rate_pct"])
                  .sort_values("readmit_rate_pct", ascending=True))

    fig, ax = plt.subplots(figsize=(9, max(5, len(rates_df) * 0.45)))
    colors = ["#F44336" if v > 13 else "#FF9800" if v > 11 else "#4CAF50"
              for v in rates_df["readmit_rate_pct"]]
    ax.barh(rates_df.index, rates_df["readmit_rate_pct"],
            color=colors, edgecolor="white")
    for i, (idx, row) in enumerate(rates_df.iterrows()):
        ax.text(row["readmit_rate_pct"] + 0.1, i,
                f"{row['readmit_rate_pct']:.1f}%", va="center", fontsize=9)
    ax.set_title("30-Day Readmission Rate by Medication Change",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Readmission Rate (%) — patients with dose changed", fontsize=LABEL_FS)
    ax.set_xlim(0, rates_df["readmit_rate_pct"].max() * 1.2)
    _save(fig, os.path.join(out_dir, "eda_top_medications.png"))


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Starting EDA — {df.shape[0]:,} rows, {df.shape[1]} columns")

    plot_class_distribution(df, out_dir)
    plot_missing_values(df, out_dir)
    plot_age_distribution(df, out_dir)
    plot_numeric_distributions(df, out_dir)
    plot_correlation_heatmap(df, out_dir)
    plot_readmit_by_time(df, out_dir)
    plot_readmit_by_inpatient(df, out_dir)
    plot_top_medications(df, out_dir)

    logger.info("EDA complete — all figures saved.")


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.ingestion.load_data import load_full_dataframe
    df = load_full_dataframe(cfg["paths"]["raw_data"])
    run_eda(df, cfg["paths"]["figures"])
    print("\n✅ Phase 2 complete. Check reports/figures/ for all EDA plots.")
