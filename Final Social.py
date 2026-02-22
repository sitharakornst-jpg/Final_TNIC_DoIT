# -*- coding: utf-8 -*-

import os
import time
import importlib
import numpy as np
import pandas as pd
import xgboost as xgb

try:
    tqdm = importlib.import_module('tqdm.auto').tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


global_start = time.time()

# 1. Setup Directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR).lower() == 'xgboost' else SCRIPT_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'XGBoost')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'[1/7] Output directory initialized at {OUTPUT_DIR}')

# 2. Load Data
print('[2/7] Loading datasets...')
files = [
    'price_cost.csv', 'sales_history.csv', 'inventory.csv', 'competitor_prices.csv',
    'sku_master.csv', 'store_master.csv', 'promo_log.csv', 'calendar_weather.csv',
    'XEL.csv', 'sample_submission.csv'
]

dfs = {}
load_start = time.time()
for file_name in tqdm(files, desc='Loading Files'):
    dfs[file_name.split('.')[0]] = pd.read_csv(os.path.join(BASE_DIR, file_name))
print(f'   - Data loaded in {time.time() - load_start:.2f}s')

# 3. Pre-processing
print('[3/7] Pre-processing data...')
df_sales = dfs['sales_history'].copy()
df_sales['date'] = pd.to_datetime(df_sales['date'])

df_template = dfs['sample_submission'].copy()
df_template['date'] = pd.to_datetime(df_template['date'], format='%d/%m/%Y')

df_price_cost = dfs['price_cost'].copy()
df_price_cost['vat_rate'] = df_price_cost['vat_rate'].fillna(0.07)

df_opt = df_template.merge(
    df_price_cost[['sku_id', 'regular_price', 'unit_cost', 'vat_rate']], on='sku_id', how='left'
)
df_opt = df_opt.merge(dfs['sku_master'], on='sku_id', how='left')
df_opt = df_opt.merge(dfs['store_master'], on='store_id', how='left')

# 4. Feature Engineering
print('[4/7] Engineering features...')
daily_sales = df_sales.groupby(['date', 'sku_id'])['qty'].sum().reset_index()
daily_sales['qty_lag_7'] = daily_sales.groupby('sku_id')['qty'].shift(7)
daily_sales['qty_rmean_7'] = daily_sales.groupby('sku_id')['qty'].transform(lambda x: x.shift(7).rolling(7).mean())

# 14-day demand base
last_sales_date = df_sales['date'].max()
cutoff_14d = last_sales_date - pd.Timedelta(days=13)
recent_sales_14d = df_sales[df_sales['date'] >= cutoff_14d]
avg_demand_14d = (
    recent_sales_14d
    .groupby(['store_id', 'sku_id'])['qty']
    .mean()
    .reset_index()
    .rename(columns={'qty': 'avg_daily_demand_14d'})
)
df_opt = df_opt.merge(avg_demand_14d, on=['store_id', 'sku_id'], how='left')
df_opt['avg_daily_demand_14d'] = df_opt['avg_daily_demand_14d'].fillna(0)

# Competitor and inventory features
latest_comp = dfs['competitor_prices'].copy()
latest_comp['date'] = pd.to_datetime(latest_comp['date'])
latest_comp = latest_comp.sort_values('date').groupby('sku_id').tail(1)[['sku_id', 'comp_price']]
df_opt = df_opt.merge(latest_comp, on='sku_id', how='left')
df_opt['comp_price'] = df_opt['comp_price'].fillna(df_opt['regular_price'])

latest_inv = dfs['inventory'].groupby('sku_id')['on_hand'].sum().reset_index()
df_opt = df_opt.merge(latest_inv, on='sku_id', how='left').fillna({'on_hand': 0})

# 14-day stockout model (rule-based)
df_opt['expected_units_14d'] = df_opt['avg_daily_demand_14d'] * 14
df_opt['stockout_risk_14d'] = (df_opt['on_hand'] < df_opt['expected_units_14d']).astype(int)

# Latest observed selling price (for day-to-day fluctuation control)
latest_selling_price = (
    df_sales.sort_values('date')
    .groupby(['store_id', 'sku_id'])
    .tail(1)[['store_id', 'sku_id', 'price_paid']]
    .rename(columns={'price_paid': 'last_price_paid'})
)
df_opt = df_opt.merge(latest_selling_price, on=['store_id', 'sku_id'], how='left')
df_opt['last_price_paid'] = df_opt['last_price_paid'].fillna(df_opt['regular_price'])

# Social category tag (essential products get stronger affordability protection)
essential_keywords = r'water|milk|rice|egg|oil|noodle|baby|diaper|medicine|health'
essential_mask = (
    df_opt['category'].astype(str).str.lower().str.contains(essential_keywords, regex=True) |
    df_opt['subcategory'].astype(str).str.lower().str.contains(essential_keywords, regex=True)
)
df_opt['is_essential'] = essential_mask.astype(int)

# 5. Train XGBoost (kept for compatibility with existing workflow)
print('[5/7] Training XGBoost model...')
train_start = time.time()
df_train = daily_sales.dropna().copy()
X_train = df_train[['sku_id', 'qty_lag_7', 'qty_rmean_7']]
y_train = df_train['qty']
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)
print(f'   - Model trained in {time.time() - train_start:.2f}s')

# 6. Optimization with social-balanced objective
print('[6/7] Running Price Optimization...')
opt_start = time.time()

MAX_DAILY_CHANGE_PCT = 0.10
ESSENTIAL_MAX_GAP_PCT = 0.02  # essential SKUs can be at most +2% above competitor


deltas = np.linspace(-0.20, 0.30, 11)
reg_prices = df_opt['regular_price'].values[:, np.newaxis]
costs = df_opt['unit_cost'].values[:, np.newaxis]
comp_prices = df_opt['comp_price'].values[:, np.newaxis]
vat_multiplier = (1 + df_opt['vat_rate'].values[:, np.newaxis])
expected_units_daily = df_opt['avg_daily_demand_14d'].values[:, np.newaxis]
last_prices = df_opt['last_price_paid'].values[:, np.newaxis]
essential_factor = df_opt['is_essential'].values[:, np.newaxis]

candidates = reg_prices * (1 + deltas)


def apply_rounding(arr):
    base = np.floor(arr)
    c1, c2, c3 = base + 0.0, base + 0.5, base + 0.9
    d1, d2, d3 = np.abs(arr - c1), np.abs(arr - c2), np.abs(arr - c3)
    return np.where(d1 < d2, np.where(d1 < d3, c1, c3), np.where(d2 < d3, c2, c3))


candidates = apply_rounding(candidates)

# Business rules
min_markdown_price = reg_prices * 0.70
min_cost_vat_price = costs * vat_multiplier
daily_floor = last_prices * (1 - MAX_DAILY_CHANGE_PCT)
daily_ceiling = last_prices * (1 + MAX_DAILY_CHANGE_PCT)

rule_markdown = candidates >= min_markdown_price
rule_min_price = candidates >= min_cost_vat_price
rule_daily_change = (candidates >= daily_floor) & (candidates <= daily_ceiling)

# Social guardrail: essential products should not be much higher than competitor
essential_cap = np.where(essential_factor == 1, comp_prices * (1 + ESSENTIAL_MAX_GAP_PCT), np.inf)
rule_essential_fair = candidates <= essential_cap

feasible_mask = rule_markdown & rule_min_price & rule_daily_change & rule_essential_fair

candidates_before_vat = candidates / vat_multiplier
valid_margin_mask = candidates_before_vat > costs

gross_profit_per_unit = candidates_before_vat - costs
gross_profit_total = gross_profit_per_unit * expected_units_daily

# Penalties
comp_gap_up = np.maximum(0, candidates - comp_prices)
affordability_penalty = comp_gap_up * np.where(essential_factor == 1, 1.20, 0.60)
instability_penalty = np.abs(candidates - last_prices) * 0.30
stockout_penalty = np.where(df_opt['stockout_risk_14d'].values[:, np.newaxis] == 1,
                            np.maximum(0, candidates - reg_prices) * 0.15,
                            0)

scores = np.where(
    feasible_mask & valid_margin_mask,
    gross_profit_total - affordability_penalty - instability_penalty - stockout_penalty,
    -1e12
)

best_indices = np.argmax(scores, axis=1)
df_opt['proposed_price'] = candidates[np.arange(len(df_opt)), best_indices]

# Fallback if no feasible candidate
no_feasible = ~np.any(feasible_mask & valid_margin_mask, axis=1)
if np.any(no_feasible):
    fallback_floor = np.maximum.reduce([
        df_opt['regular_price'].values * 0.70,
        df_opt['unit_cost'].values * (1 + df_opt['vat_rate'].values),
        df_opt['last_price_paid'].values * (1 - MAX_DAILY_CHANGE_PCT),
    ])
    fallback_cap = np.minimum(
        df_opt['last_price_paid'].values * (1 + MAX_DAILY_CHANGE_PCT),
        np.where(df_opt['is_essential'].values == 1,
                 df_opt['comp_price'].values * (1 + ESSENTIAL_MAX_GAP_PCT),
                 np.inf)
    )

    fallback_candidates = np.column_stack([
        np.floor(fallback_floor) + 0.0,
        np.floor(fallback_floor) + 0.5,
        np.floor(fallback_floor) + 0.9,
        np.ceil(fallback_floor) + 0.0,
        np.ceil(fallback_floor) + 0.5,
        np.ceil(fallback_floor) + 0.9,
    ])
    fallback_ok = (fallback_candidates >= fallback_floor[:, None]) & (fallback_candidates <= fallback_cap[:, None])
    fallback_candidates = np.where(fallback_ok, fallback_candidates, np.inf)
    chosen_fallback = np.min(fallback_candidates, axis=1)

    unresolved = np.isinf(chosen_fallback)
    if np.any(unresolved):
        fallback_unresolved = apply_rounding(fallback_floor[unresolved][:, np.newaxis]).ravel()
        chosen_fallback[unresolved] = fallback_unresolved

    df_opt.loc[no_feasible, 'proposed_price'] = chosen_fallback[no_feasible]

# Final hard enforcement for requested rules
proposed = df_opt['proposed_price'].values

# Rule ending must be 0.00, 0.50, 0.90
proposed = apply_rounding(proposed[:, np.newaxis]).ravel()

# Essential fairness must hold (essential SKUs <= competitor * (1 + cap))
essential_cap_series = np.where(
    df_opt['is_essential'].values == 1,
    df_opt['comp_price'].values * (1 + ESSENTIAL_MAX_GAP_PCT),
    np.inf
)
violating_essential = (df_opt['is_essential'].values == 1) & (proposed > essential_cap_series)
if np.any(violating_essential):
    cap_vals = essential_cap_series[violating_essential]
    cap_base = np.floor(cap_vals)
    cap_90 = cap_base + 0.9
    cap_50 = cap_base + 0.5
    cap_00 = cap_base + 0.0
    rounded_down = np.where(cap_90 <= cap_vals, cap_90, np.where(cap_50 <= cap_vals, cap_50, cap_00))
    proposed[violating_essential] = rounded_down

df_opt['proposed_price'] = proposed

# Final metrics
df_opt['proposed_price_before_vat'] = df_opt['proposed_price'] / (1 + df_opt['vat_rate'])
df_opt['gross_profit_per_unit'] = df_opt['proposed_price_before_vat'] - df_opt['unit_cost']
df_opt['expected_unit_sold'] = df_opt['avg_daily_demand_14d']
df_opt['expected_gross_profit'] = df_opt['gross_profit_per_unit'] * df_opt['expected_unit_sold']
df_opt['price_gap_vs_comp'] = df_opt['proposed_price'] - df_opt['comp_price']

frac = df_opt['proposed_price'] - np.floor(df_opt['proposed_price'])
df_opt['rule_allowed_ending'] = np.isclose(frac, 0.0, atol=1e-6) | np.isclose(frac, 0.5, atol=1e-6) | np.isclose(frac, 0.9, atol=1e-6)
df_opt['rule_markdown_30pct'] = df_opt['proposed_price'] >= (df_opt['regular_price'] * 0.70)
df_opt['rule_min_price_cost_vat'] = df_opt['proposed_price'] >= (df_opt['unit_cost'] * (1 + df_opt['vat_rate']))
df_opt['rule_daily_fluctuation'] = (
    (df_opt['proposed_price'] >= (df_opt['last_price_paid'] * (1 - MAX_DAILY_CHANGE_PCT))) &
    (df_opt['proposed_price'] <= (df_opt['last_price_paid'] * (1 + MAX_DAILY_CHANGE_PCT)))
)
df_opt['rule_essential_fair'] = np.where(
    df_opt['is_essential'] == 1,
    df_opt['proposed_price'] <= df_opt['comp_price'] * (1 + ESSENTIAL_MAX_GAP_PCT),
    True
)

print(f'   - Optimization completed in {time.time() - opt_start:.4f}s')

# 7. Save Outputs
print('[7/7] Saving results...')
run_tag = int(time.time())
output_path = os.path.join(OUTPUT_DIR, f'final_submission_social_balanced_v1_{run_tag}.csv')
submission_df = df_opt[['ID', 'store_id', 'sku_id', 'date', 'proposed_price']].copy()
submission_df['date'] = submission_df['date'].dt.strftime('%d/%m/%Y')
submission_df.to_csv(output_path, index=False)

analysis_output_path = os.path.join(OUTPUT_DIR, f'final_submission_analysis_social_balanced_v1_{run_tag}.csv')
analysis_cols = [
    'ID', 'store_id', 'sku_id', 'date', 'category', 'subcategory', 'is_essential',
    'regular_price', 'comp_price', 'proposed_price', 'price_gap_vs_comp',
    'proposed_price_before_vat', 'unit_cost', 'vat_rate', 'last_price_paid',
    'expected_unit_sold', 'gross_profit_per_unit', 'expected_gross_profit',
    'on_hand', 'expected_units_14d', 'stockout_risk_14d',
    'rule_allowed_ending', 'rule_markdown_30pct', 'rule_min_price_cost_vat',
    'rule_daily_fluctuation', 'rule_essential_fair'
]
df_opt[analysis_cols].to_csv(analysis_output_path, index=False)

# Summary
n_rows = len(df_opt)
over_comp_count = int((df_opt['price_gap_vs_comp'] > 0).sum())
essential_over_comp_count = int(((df_opt['is_essential'] == 1) & (df_opt['price_gap_vs_comp'] > 0)).sum())

print('\n' + '=' * 30)
print(f'Done! Final submission saved to: {output_path}')
print(f'Analysis file saved to: {analysis_output_path}')
print(f'Expected total gross profit: {df_opt["expected_gross_profit"].sum():,.2f}')
print(f'Over competitor rows (all): {over_comp_count}/{n_rows}')
print(f'Over competitor rows (essential): {essential_over_comp_count}/{int((df_opt["is_essential"] == 1).sum())}')
print(f'Rule pass - allowed ending: {int(df_opt["rule_allowed_ending"].sum())}/{n_rows}')
print(f'Rule pass - markdown <= 30%: {int(df_opt["rule_markdown_30pct"].sum())}/{n_rows}')
print(f'Rule pass - min price cost+VAT: {int(df_opt["rule_min_price_cost_vat"].sum())}/{n_rows}')
print(f'Rule pass - daily fluctuation: {int(df_opt["rule_daily_fluctuation"].sum())}/{n_rows}')
print(f'Rule pass - essential fairness: {int(df_opt["rule_essential_fair"].sum())}/{n_rows}')
print(f'Total Execution Time: {time.time() - global_start:.2f}s')
print('=' * 30)
print(submission_df.head(10).to_string(index=False))
