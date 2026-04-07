"""
Tests for Data Analysis Foundations — Analytics Pipeline
Author: Anandi Mahure | MSc Data Science, University of Bath
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis import (
    load_and_clean,
    descriptive_stats,
    detect_outliers_iqr,
    detect_outliers_zscore,
    ttest_spending_by_churn,
    ttest_income_by_gender,
    segment_customers,
)

# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def df():
    return load_and_clean()


# ── 1. Schema & Data Quality ──────────────────────────────────

class TestSchema:
    REQUIRED_COLUMNS = [
        'customer_id', 'age', 'gender', 'region', 'income',
        'spending_score', 'transactions_per_year', 'total_spend',
        'preferred_category', 'loyalty_years', 'churn_risk'
    ]

    def test_required_columns_exist(self, df):
        for col in self.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_after_imputation(self, df):
        assert df['income'].isna().sum() == 0
        assert df['spending_score'].isna().sum() == 0

    def test_no_duplicate_customer_ids(self, df):
        assert df['customer_id'].duplicated().sum() == 0

    def test_row_count(self, df):
        assert len(df) == 500

    def test_age_in_valid_range(self, df):
        assert df['age'].between(18, 70).all()

    def test_spending_score_in_valid_range(self, df):
        assert df['spending_score'].between(0, 100).all()

    def test_income_positive(self, df):
        assert (df['income'] > 0).all()

    def test_total_spend_positive(self, df):
        assert (df['total_spend'] > 0).all()


# ── 2. Business Logic ────────────────────────────────────────

class TestBusinessLogic:
    VALID_GENDERS = {'Male', 'Female', 'Non-Binary'}
    VALID_CHURN = {'Low', 'Medium', 'High'}
    VALID_REGIONS = {'London', 'Manchester', 'Birmingham', 'Leeds', 'Bristol'}
    VALID_CATEGORIES = {'Electronics', 'Clothing', 'Grocery', 'Health & Beauty', 'Sports'}

    def test_valid_gender_values(self, df):
        assert set(df['gender'].unique()).issubset(self.VALID_GENDERS)

    def test_valid_churn_risk_values(self, df):
        assert set(df['churn_risk'].unique()).issubset(self.VALID_CHURN)

    def test_valid_regions(self, df):
        assert set(df['region'].unique()).issubset(self.VALID_REGIONS)

    def test_valid_categories(self, df):
        assert set(df['preferred_category'].unique()).issubset(self.VALID_CATEGORIES)

    def test_loyalty_years_non_negative(self, df):
        assert (df['loyalty_years'] >= 0).all()

    def test_transactions_positive(self, df):
        assert (df['transactions_per_year'] > 0).all()

    def test_london_largest_region(self, df):
        region_counts = df['region'].value_counts()
        assert region_counts.index[0] == 'London'

    def test_all_churn_risk_levels_present(self, df):
        assert set(df['churn_risk'].unique()) == self.VALID_CHURN


# ── 3. Descriptive Statistics ────────────────────────────────

class TestDescriptiveStats:
    def test_stats_returns_dataframe(self, df):
        result = descriptive_stats(df)
        assert isinstance(result, pd.DataFrame)

    def test_stats_includes_key_columns(self, df):
        result = descriptive_stats(df)
        assert 'income' in result.columns
        assert 'spending_score' in result.columns
        assert 'age' in result.columns

    def test_mean_spending_score_in_range(self, df):
        assert 40 <= df['spending_score'].mean() <= 65

    def test_mean_age_in_range(self, df):
        assert 28 <= df['age'].mean() <= 42

    def test_mean_income_reasonable(self, df):
        assert 30000 <= df['income'].mean() <= 60000


# ── 4. Outlier Detection ────────────────────────────────────

class TestOutlierDetection:
    def test_iqr_returns_boolean_series(self, df):
        result = detect_outliers_iqr(df['total_spend'])
        assert result.dtype == bool

    def test_zscore_returns_boolean_array(self, df):
        result = detect_outliers_zscore(df['total_spend'])
        assert result.dtype == bool

    def test_iqr_outlier_count_reasonable(self, df):
        result = detect_outliers_iqr(df['total_spend'])
        pct = result.sum() / len(df)
        assert 0.0 < pct < 0.15

    def test_zscore_fewer_than_iqr(self, df):
        iqr_count = detect_outliers_iqr(df['total_spend']).sum()
        z_count = detect_outliers_zscore(df['total_spend']).sum()
        assert z_count <= iqr_count


# ── 5. Hypothesis Testing ────────────────────────────────────

class TestHypothesisTesting:
    def test_churn_ttest_returns_floats(self, df):
        t, p = ttest_spending_by_churn(df)
        assert isinstance(t, float)
        assert isinstance(p, float)

    def test_churn_ttest_significant(self, df):
        _, p = ttest_spending_by_churn(df)
        assert p < 0.05, "Expected significant difference in spending by churn risk"

    def test_gender_ttest_returns_floats(self, df):
        t, p = ttest_income_by_gender(df)
        assert isinstance(t, float)
        assert isinstance(p, float)

    def test_pvalue_in_valid_range(self, df):
        _, p1 = ttest_spending_by_churn(df)
        _, p2 = ttest_income_by_gender(df)
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1


# ── 6. Customer Segmentation ────────────────────────────────

class TestSegmentation:
    def test_spending_tier_column_created(self, df):
        result = segment_customers(df)
        assert 'spending_tier' in result.columns

    def test_all_tiers_present(self, df):
        result = segment_customers(df)
        tiers = set(result['spending_tier'].dropna().unique())
        assert 'Low Spender' in tiers
        assert 'Mid Spender' in tiers
        assert 'High Spender' in tiers

    def test_no_nan_in_tiers(self, df):
        result = segment_customers(df)
        assert result['spending_tier'].isna().sum() == 0

    def test_low_spenders_have_lower_scores(self, df):
        result = segment_customers(df)
        low_avg = result[result['spending_tier'] == 'Low Spender']['spending_score'].mean()
        high_avg = result[result['spending_tier'] == 'High Spender']['spending_score'].mean()
        assert low_avg < high_avg


# ── 7. Chart Output ────────────────────────────────────────

class TestChartOutput:
    EXPECTED_CHARTS = [
        '01_spending_distribution.png',
        '02_income_vs_spending.png',
        '03_regional_breakdown.png',
        '04_churn_by_tier.png',
        '05_category_overview.png',
    ]

    def test_charts_directory_exists(self):
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        assert os.path.isdir(charts_dir), "charts/ directory not found"

    def test_all_charts_present(self):
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        if not os.path.isdir(charts_dir):
            pytest.skip("charts/ not found")
        for chart in self.EXPECTED_CHARTS:
            path = os.path.join(charts_dir, chart)
            assert os.path.exists(path), f"Missing chart: {chart}"

    def test_charts_non_empty(self):
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        if not os.path.isdir(charts_dir):
            pytest.skip("charts/ not found")
        for chart in self.EXPECTED_CHARTS:
            path = os.path.join(charts_dir, chart)
            if os.path.exists(path):
                assert os.path.getsize(path) > 10_000, f"Chart too small: {chart}"
