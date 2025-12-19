import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# STRICT THRESHOLD: Problem is "Solved" only if pass_rate >= 0.95
SUCCESS_THRESHOLD = 0.95 
OUTPUT_FILENAME = 'dataset_training_mbpp.csv'
# ---------------------

# 1. Load the Data
# Ensure these filenames match your directory
df_4b = pd.read_csv('extracted_confidences_qwen4B_it_50problems(Sheet1).csv')
df_32b = pd.read_csv('extracted_confidences_qwen4B_32B_it_50_problems (2)(Sheet1).csv')

# 2. Clean Data
def clean_df(df, model_size):
    cols = ['bottom_window_confidence', 'least_grouped_confidence', 
            'mean_confidence', 'tail_confidence', 'pass_rate']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['current_model_size'] = model_size
    return df.dropna(subset=cols)

df_4b = clean_df(df_4b, 4)
df_32b = clean_df(df_32b, 32)

# 3. Establish Ground Truth
# We determine if a problem is solvable (>= 0.95) by each model based on specific runs
ground_truth = pd.DataFrame()
all_pids = pd.concat([df_4b['problem_id'], df_32b['problem_id']]).unique()
ground_truth['problem_id'] = all_pids

# Calculate max pass rate achieved by each model per problem
max_pass_4b = df_4b.groupby('problem_id')['pass_rate'].max()
max_pass_32b = df_32b.groupby('problem_id')['pass_rate'].max()

ground_truth = ground_truth.set_index('problem_id')
ground_truth['max_pass_4b'] = max_pass_4b
ground_truth['max_pass_32b'] = max_pass_32b

# Boolean Solvability based on strict threshold
ground_truth['can_4b_solve'] = ground_truth['max_pass_4b'] >= SUCCESS_THRESHOLD
ground_truth['can_32b_solve'] = ground_truth['max_pass_32b'] >= SUCCESS_THRESHOLD
ground_truth.fillna(False, inplace=True)

# 4. Generate Labels (The "Router" Logic)
combined_data = []

def get_target_class(row):
    pid = row['problem_id']
    if pid not in ground_truth.index: return None 
    
    truth = ground_truth.loc[pid]
    
    # Is the CURRENT run a success?
    current_run_success = row['pass_rate'] >= SUCCESS_THRESHOLD
    model_size = row['current_model_size']
    
    # TARGET 0 = Use 4B (Default/Stay)
    # TARGET 1 = Use 32B (Switch/Upgrade)
    
    if model_size == 4:
        if current_run_success:
            return 0 # 4B worked perfectly. Stay.
        else:
            # 4B failed (< 0.95). Can 32B solve it strictly (>= 0.95)?
            if truth['can_32b_solve']:
                return 1 # Yes, switch.
            else:
                return 0 # No, 32B also fails/gets partial credit. Don't waste compute.
                
    elif model_size == 32:
        if current_run_success:
            if truth['can_4b_solve']:
                return 0 # 4B could have done this. Switch Down.
            else:
                return 1 # Only 32B can do this. Stay.
        else:
            return 0 # 32B failed. Revert to base.

    return 0

for df in [df_4b, df_32b]:
    for index, row in df.iterrows():
        target = get_target_class(row)
        if target is not None:
            row_data = row.to_dict()
            row_data['target_class'] = target
            combined_data.append(row_data)

# 5. Save
final_df = pd.DataFrame(combined_data)
features = ['current_model_size', 'mean_confidence', 'bottom_window_confidence', 
            'tail_confidence', 'least_grouped_confidence', 'target_class']

dataset_for_training = final_df[features]
dataset_for_training.to_csv(OUTPUT_FILENAME, index=False)

print(f"Dataset created: {OUTPUT_FILENAME}")
print(f"Logic: Switch to 32B only if 4B < {SUCCESS_THRESHOLD} AND 32B >= {SUCCESS_THRESHOLD}")
print(dataset_for_training['target_class'].value_counts())
