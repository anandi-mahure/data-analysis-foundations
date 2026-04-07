"""
Data Analysis Foundations — Python Analytics Pipeline
Author: Anandi Mahure | MSc Data Science, University of Bath (Dean's Award 2025)
Description: End-to-end customer analytics pipeline demonstrating statistical
analysis, EDA, outlier detection, segmentation and hypothesis testing
on a retail customer behaviour dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats

# ── PATHS ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'customer_analytics.csv')
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── PALETTE ───────────────────────────────────────────────────
C1 = '#1F4E79'
C2 = '#2E75B6'
C3 = '#5BA3D9'
C4 = '#D6E4F0'
RED = '#C00000'
GRN = '#2D6A2D'
WHT = '#FFFFFF'
GRY = '#F7F9FC'


def watermark(ax):
    ax.annotate('Anandi Mahure | MSc Data Science, University of Bath',
                xy=(1, 0), xycoords='axes fraction',
                fontsize=7, color='#AAAAAA', ha='right', va='bottom')


def base_style(fig, ax):
    fig.patch.set_facecolor(WHT)
    ax.set_facecolor(GRY)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#CCCCCC')
    ax.tick_params(colors='#555555', labelsize=9)


# ═══════════════════════════════════════════════════════════════
# LOAD & CLEAN DATA
# ═══════════════════════════════════════════════════════════════

def load_and_clean(path=DATA_PATH):
    """Load CSV, parse types, impute missing values."""
    df = pd.read_csv(path)
    df['income'] = pd.to_numeric(df['income'], errors='coerce')
    df['spending_score'] = pd.to_numeric(df['spending_score'], errors='coerce')
    df['income'] = df['income'].fillna(df['income'].median())
    df['spending_score'] = df['spending_score'].fillna(df['spending_score'].median())
    return df


# ═══════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════

def descriptive_stats(df):
    """Return summary statistics for numeric columns."""
    numeric = df.select_dtypes(include='number')
    return numeric.describe()


# ═══════════════════════════════════════════════════════════════
# OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_outliers_iqr(series):
    """Return boolean mask of outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def detect_outliers_zscore(series, threshold=3):
    """Return boolean mask of outliers using Z-score method."""
    z = np.abs(stats.zscore(series.dropna()))
    return z > threshold


# ═══════════════════════════════════════════════════════════════
# HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════

def ttest_spending_by_churn(df):
    """
    H0: Mean spending score is equal for High vs Low churn risk customers.
    H1: There is a significant difference in spending score by churn risk.
    """
    high = df[df['churn_risk'] == 'High']['spending_score']
    low = df[df['churn_risk'] == 'Low']['spending_score']
    t_stat, p_value = stats.ttest_ind(high, low)
    return t_stat, p_value


def ttest_income_by_gender(df):
    """
    H0: Mean income is equal for Male vs Female customers.
    H1: There is a significant difference in income by gender.
    """
    male = df[df['gender'] == 'Male']['income']
    female = df[df['gender'] == 'Female']['income']
    t_stat, p_value = stats.ttest_ind(male, female)
    return t_stat, p_value


# ═══════════════════════════════════════════════════════════════
# CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def segment_customers(df):
    """Assign spending tier based on spending score."""
    df = df.copy()
    df['spending_tier'] = pd.cut(
        df['spending_score'],
        bins=[0, 33, 66, 100],
        labels=['Low Spender', 'Mid Spender', 'High Spender']
    )
    return df


# ═══════════════════════════════════════════════════════════════
# CHART 1 — Spending Score Distribution
# ═══════════════════════════════════════════════════════════════

def chart_spending_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    base_style(fig, ax)

    ax.hist(df['spending_score'], bins=25, color=C2, edgecolor=WHT,
            linewidth=0.8, zorder=3, alpha=0.85)

    mean_val = df['spending_score'].mean()
    median_val = df['spending_score'].median()
    ax.axvline(mean_val, color=C1, linestyle='--', linewidth=1.8,
               label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color=RED, linestyle='--', linewidth=1.8,
               label=f'Median: {median_val:.1f}')

    ax.set_xlabel('Spending Score', fontsize=10)
    ax.set_ylabel('Number of Customers', fontsize=10)
    ax.set_title('Spending Score Distribution',
                 fontsize=14, fontweight='bold', color=C1, pad=16, loc='left')
    ax.annotate('Normally distributed with slight right skew — '
                'majority of customers score between 30–70.',
                xy=(0, 1.04), xycoords='axes fraction',
                fontsize=9, color='#666666', style='italic')
    ax.legend(fontsize=9, framealpha=0)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.7, zorder=0)
    watermark(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '01_spending_distribution.png'),
                dpi=180, bbox_inches='tight', facecolor=WHT)
    plt.close()
    print("Chart 1 done")


# ═══════════════════════════════════════════════════════════════
# CHART 2 — Income vs Spending Score (Scatter + Regression)
# ═══════════════════════════════════════════════════════════════

def chart_income_vs_spending(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, ax)

    colors = df['churn_risk'].map({'Low': GRN, 'Medium': C2, 'High': RED})
    ax.scatter(df['income'], df['spending_score'],
               c=colors, alpha=0.5, s=28, zorder=3, edgecolors='none')

    m, b, r, p, _ = stats.linregress(df['income'], df['spending_score'])
    x_line = np.linspace(df['income'].min(), df['income'].max(), 200)
    ax.plot(x_line, m * x_line + b, color=C1, linewidth=2,
            label=f'Regression line (r={r:.2f})')

    ax.set_xlabel('Annual Income (£)', fontsize=10)
    ax.set_ylabel('Spending Score', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1000:.0f}K'))
    ax.set_title('Income vs Spending Score by Churn Risk',
                 fontsize=14, fontweight='bold', color=C1, pad=16, loc='left')
    ax.annotate('Weak positive correlation between income and spending score. '
                'High-churn customers cluster at low spending regardless of income.',
                xy=(0, 1.04), xycoords='axes fraction',
                fontsize=9, color='#666666', style='italic')

    legend_patches = [
        mpatches.Patch(color=GRN, label='Low Churn Risk'),
        mpatches.Patch(color=C2, label='Medium Churn Risk'),
        mpatches.Patch(color=RED, label='High Churn Risk'),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches, fontsize=8, framealpha=0.9,
              edgecolor='#CCCCCC')
    ax.grid(color='#DDDDDD', linewidth=0.7, zorder=0)
    watermark(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '02_income_vs_spending.png'),
                dpi=180, bbox_inches='tight', facecolor=WHT)
    plt.close()
    print("Chart 2 done")


# ═══════════════════════════════════════════════════════════════
# CHART 3 — Regional Customer Breakdown
# ═══════════════════════════════════════════════════════════════

def chart_regional_breakdown(df):
    regional = df.groupby('region').agg(
        count=('customer_id', 'count'),
        avg_spend=('spending_score', 'mean'),
        avg_income=('income', 'mean')
    ).sort_values('count', ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(WHT)

    # Left — customer count
    ax = axes[0]
    ax.set_facecolor(GRY)
    bar_colors = [C1, C2, C3, C4, '#B8D4EA']
    bars = ax.barh(regional['region'], regional['count'],
                   color=bar_colors, edgecolor=WHT, height=0.45, zorder=3)
    for bar, val in zip(bars, regional['count']):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', fontsize=10,
                fontweight='bold', color=C1)
    ax.set_xlabel('Number of Customers', fontsize=10)
    ax.set_title('Customers by Region', fontsize=11,
                 fontweight='bold', color=C1, pad=10)
    ax.grid(axis='x', color='#DDDDDD', linewidth=0.7, zorder=0)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#CCCCCC')
    ax.tick_params(colors='#555555', labelsize=10)

    # Right — avg spending score
    ax = axes[1]
    ax.set_facecolor(GRY)
    bars2 = ax.barh(regional['region'], regional['avg_spend'],
                    color=bar_colors, edgecolor=WHT, height=0.45, zorder=3)
    for bar, val in zip(bars2, regional['avg_spend']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=10,
                fontweight='bold', color=C1)
    ax.set_xlabel('Average Spending Score', fontsize=10)
    ax.set_title('Avg Spending Score by Region', fontsize=11,
                 fontweight='bold', color=C1, pad=10)
    ax.grid(axis='x', color='#DDDDDD', linewidth=0.7, zorder=0)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#CCCCCC')
    ax.tick_params(colors='#555555', labelsize=10)

    fig.suptitle('Regional Customer Analysis', fontsize=14,
                 fontweight='bold', color=C1, x=0.02, ha='left', y=1.02)
    fig.text(0.02, 0.97,
             'London leads in volume (35%). Spending scores are consistent across regions.',
             fontsize=9, color='#666666', style='italic')
    fig.text(0.99, -0.02, 'Anandi Mahure | MSc Data Science, University of Bath',
             fontsize=7, color='#AAAAAA', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '03_regional_breakdown.png'),
                dpi=180, bbox_inches='tight', facecolor=WHT)
    plt.close()
    print("Chart 3 done")


# ═══════════════════════════════════════════════════════════════
# CHART 4 — Churn Risk by Spending Tier
# ═══════════════════════════════════════════════════════════════

def chart_churn_by_tier(df):
    df = segment_customers(df)
    pivot = df.groupby(['spending_tier', 'churn_risk']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    base_style(fig, ax)

    x = np.arange(len(pivot_pct))
    width = 0.25
    risk_colors = {'Low': GRN, 'Medium': C2, 'High': RED}

    for i, risk in enumerate(['Low', 'Medium', 'High']):
        if risk in pivot_pct.columns:
            bars = ax.bar(x + i * width, pivot_pct[risk],
                          width=width, color=risk_colors[risk],
                          edgecolor=WHT, zorder=3, label=f'{risk} Churn')
            for bar, val in zip(bars, pivot_pct[risk]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.0f}%', ha='center', fontsize=8,
                        fontweight='bold', color='#333333')

    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot_pct.index, fontsize=10)
    ax.set_ylabel('% of Customers', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title('Churn Risk Distribution by Spending Tier',
                 fontsize=14, fontweight='bold', color=C1, pad=16, loc='left')
    ax.annotate('Low spenders carry >80% high churn risk. '
                'High spenders are predominantly low churn — strong retention signal.',
                xy=(0, 1.04), xycoords='axes fraction',
                fontsize=9, color='#666666', style='italic')
    ax.legend(fontsize=9, framealpha=0)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.7, zorder=0)
    watermark(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '04_churn_by_tier.png'),
                dpi=180, bbox_inches='tight', facecolor=WHT)
    plt.close()
    print("Chart 4 done")


# ═══════════════════════════════════════════════════════════════
# CHART 5 — Category Revenue & Outlier Summary
# ═══════════════════════════════════════════════════════════════

def chart_category_overview(df):
    cat = df.groupby('preferred_category').agg(
        total_spend=('total_spend', 'sum'),
        avg_spending_score=('spending_score', 'mean'),
        customer_count=('customer_id', 'count')
    ).sort_values('total_spend', ascending=False).reset_index()

    iqr_outliers = detect_outliers_iqr(df['total_spend']).sum()
    z_outliers = detect_outliers_zscore(df['total_spend']).sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(WHT)

    # Left — total spend by category
    ax = axes[0]
    ax.set_facecolor(GRY)
    cat_colors = [C1, C2, C3, C4, '#B8D4EA']
    bars = ax.bar(cat['preferred_category'], cat['total_spend'] / 1000,
                  color=cat_colors, edgecolor=WHT, width=0.45, zorder=3)
    for bar, val in zip(bars, cat['total_spend'] / 1000):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'£{val:.0f}K', ha='center', fontsize=9,
                fontweight='bold', color=C1)
    ax.set_ylabel('Total Customer Spend (£000s)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:.0f}K'))
    ax.set_title('Total Spend by Category', fontsize=11,
                 fontweight='bold', color=C1, pad=10)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.7, zorder=0)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#CCCCCC')
    ax.tick_params(colors='#555555', labelsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Right — outlier summary
    ax2 = axes[1]
    ax2.set_facecolor(GRY)
    methods = ['IQR Method', 'Z-Score Method']
    counts = [iqr_outliers, z_outliers]
    bar_c = [C2, RED]
    bars2 = ax2.bar(methods, counts, color=bar_c, edgecolor=WHT,
                    width=0.35, zorder=3)
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{int(val)} outliers\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold', color=C1)
    ax2.set_ylabel('Outliers Detected (Total Spend)', fontsize=10)
    ax2.set_ylim(0, max(counts) * 1.5)
    ax2.set_title('Outlier Detection Comparison', fontsize=11,
                  fontweight='bold', color=C1, pad=10)
    ax2.grid(axis='y', color='#DDDDDD', linewidth=0.7, zorder=0)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax2.spines[sp].set_color('#CCCCCC')
    ax2.tick_params(colors='#555555', labelsize=10)

    fig.suptitle('Category Revenue & Outlier Analysis', fontsize=14,
                 fontweight='bold', color=C1, x=0.02, ha='left', y=1.02)
    fig.text(0.02, 0.97,
             'Clothing drives highest revenue. IQR and Z-score methods align closely on outlier counts.',
             fontsize=9, color='#666666', style='italic')
    fig.text(0.99, -0.02, 'Anandi Mahure | MSc Data Science, University of Bath',
             fontsize=7, color='#AAAAAA', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, '05_category_overview.png'),
                dpi=180, bbox_inches='tight', facecolor=WHT)
    plt.close()
    print("Chart 5 done")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df = load_and_clean()
    print(f"Loaded {len(df)} customers, {df.shape[1]} features")

    stats_summary = descriptive_stats(df)
    print("\nDescriptive Statistics:")
    print(stats_summary.to_string())

    t_stat, p_val = ttest_spending_by_churn(df)
    print(f"\nHypothesis Test — Spending Score by Churn Risk:")
    print(f"  t={t_stat:.3f}, p={p_val:.4f} — "
          f"{'Significant' if p_val < 0.05 else 'Not significant'} at α=0.05")

    t_stat2, p_val2 = ttest_income_by_gender(df)
    print(f"\nHypothesis Test — Income by Gender:")
    print(f"  t={t_stat2:.3f}, p={p_val2:.4f} — "
          f"{'Significant' if p_val2 < 0.05 else 'Not significant'} at α=0.05")

    iqr_count = detect_outliers_iqr(df['total_spend']).sum()
    z_count = detect_outliers_zscore(df['total_spend']).sum()
    print(f"\nOutlier Detection (Total Spend):")
    print(f"  IQR method: {iqr_count} outliers ({iqr_count/len(df)*100:.1f}%)")
    print(f"  Z-score method: {z_count} outliers ({z_count/len(df)*100:.1f}%)")

    chart_spending_distribution(df)
    chart_income_vs_spending(df)
    chart_regional_breakdown(df)
    chart_churn_by_tier(df)
    chart_category_overview(df)

    print("\nAll 5 charts generated ✅")
