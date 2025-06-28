import os
import time
import torch
import yaml

# Import your classes
from DGCDN import DGCDN, compute_class_weights_from_dataloader
from datasets.load_DGCDN_data import ReadMIMII, ReadScenarioData
from utils.SetSeed import set_random_seed
from utils.CreateLogger import create_logger
from utils.DictObj import DictObj

# ==============================================================================
# 1. Configuration: Set the path to the checkpoint you want to test
# ==============================================================================

# --- CHOOSE WHICH CHECKPOINT TO TEST ---
# a) Example for a SCENARIO-based model (e.g., from an 's1' run)
# filename_ = r'scenario_s1_acc0.8500_auc0.9200... .pth'
# checkpoint_path = r'E:\code\myMethod-20250415\checkpoints\' + filename_

# b) Example for a SECTION-based model (e.g., from a '01' run)
filename_ = r'section01_acc0.7250_auc0.6507_pre0.7146_rec0.7267_f10.7163_20250515_160916.pth'
checkpoint_path = r'C:\Users\Malones\Desktop\DGCDN\保存的结果\\' + filename_

# Set to True if you want to generate t-SNE plots during testing
GENERATE_TSNE_PLOTS = True

# ==============================================================================
# 2. Setup and Loading (No need to edit below this line)
# ==============================================================================

print(f"Loading checkpoint: {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
configs = checkpoint['configs']
seed = checkpoint['seed']
weights = checkpoint['class_weights']

# Set the random seed for reproducibility
set_random_seed(seed)

# Determine if the checkpoint is from a scenario-based or section-based run
is_scenario_mode = str(configs.fan_section).startswith('s')

datasets_src = []
datasets_tgt = []
datasets_list = []  # For logging purposes
data_loader_class = None
loader_args = {}

if is_scenario_mode:
    # --- Logic for SCENARIO-based models ---
    print(f"Detected Scenario Mode. Scenario: {configs.fan_section}")
    scenario = configs.fan_section
    scenario_definitions = {
        # --- 原始场景 (3源, 1目标) ---
        's1': {'source': ['id_00', 'id_02', 'id_04'], 'target': ['id_06']},
        's2': {'source': ['id_00', 'id_02', 'id_06'], 'target': ['id_04']},
        's3': {'source': ['id_00', 'id_04', 'id_06'], 'target': ['id_02']},
        's4': {'source': ['id_02', 'id_04', 'id_06'], 'target': ['id_00']},
        # --- 新增场景 (2源, 2目标) ---
        's5': {'source': ['id_00', 'id_02'], 'target': ['id_04', 'id_06']},
        's6': {'source': ['id_00', 'id_04'], 'target': ['id_02', 'id_06']},
        's7': {'source': ['id_00', 'id_06'], 'target': ['id_02', 'id_04']},
        's8': {'source': ['id_02', 'id_04'], 'target': ['id_00', 'id_06']},
        's9': {'source': ['id_02', 'id_06'], 'target': ['id_00', 'id_04']},
        's10': {'source': ['id_04', 'id_06'], 'target': ['id_00', 'id_02']},
        # --- 新增场景 (1源, 3目标) ---
        's11': {'source': ['id_00'], 'target': ['id_02', 'id_04', 'id_06']},
        's12': {'source': ['id_02'], 'target': ['id_00', 'id_04', 'id_06']},
        's13': {'source': ['id_04'], 'target': ['id_00', 'id_02', 'id_06']},
        's14': {'source': ['id_06'], 'target': ['id_00', 'id_02', 'id_04']},
    }
    if scenario not in scenario_definitions:
        raise ValueError(f"Unknown scenario '{scenario}' in loaded config.")

    datasets_src = scenario_definitions[scenario]['source']
    datasets_tgt = scenario_definitions[scenario]['target']
    datasets_list = datasets_src + datasets_tgt
    data_loader_class = ReadScenarioData
    # Arguments for the loader will be (scenario, domain_id, ...)
    # We will pass scenario as the first argument to the loader class
    loader_args = {'scenario': scenario}

else:
    # --- Logic for SECTION-based models ---
    print(f"Detected Section Mode. Section: {configs.fan_section}")
    section = str(configs.fan_section).zfill(2)
    if section == '00':
        datasets_list = ['W', 'X', 'Y', 'Z']
    elif section == '01':
        datasets_list = ['A', 'B', 'C']
    elif section == '02':
        datasets_list = ['L1', 'L2', 'L3', 'L4']
    else:
        raise ValueError(f"Unknown section '{section}' in loaded config.")

    # In testing, we often test the target domain the model was trained for.
    # Assuming the first domain was the target during the saved run.
    # You might need to adjust `tgt_idx` if you want to test against a different domain.
    tgt_idx = [0]
    src_idx = [i for i in range(len(datasets_list)) if i not in tgt_idx]

    datasets_tgt = [datasets_list[i] for i in tgt_idx]
    datasets_src = [datasets_list[i] for i in src_idx]
    data_loader_class = ReadMIMII
    # Arguments for the loader will be (domain, seed, section, ...)
    # We will pass section as a keyword argument
    loader_args = {'section': section}

# Update configs with the determined source/target domains
configs.datasets_tgt = datasets_tgt
configs.datasets_src = datasets_src
configs.t_sne = GENERATE_TSNE_PLOTS

# Initialize the model with loaded configs
model = DGCDN(configs, seed, weights).to(device)

# Load the model's state dictionaries
model.encoder_m.load_state_dict(checkpoint['encoder_m'])
model.encoder_h.load_state_dict(checkpoint['encoder_h'])
model.decoder.load_state_dict(checkpoint['decoder'])
model.classifer.load_state_dict(checkpoint['classifier'])
model.optimizer.load_state_dict(checkpoint['optimizer'])
if 'attention' in checkpoint:
    model.attention.load_state_dict(checkpoint['attention'])
else:
    print("Warning: 'attention' module not found in checkpoint. Skipping.")

print("\nModel successfully loaded from checkpoint.")

# Create data loaders using the determined class and arguments
# The first argument to the loader class is always the domain name (e.g., 'W' or 'id_00')
datasets_object_src = [data_loader_class(domain, seed, configs=configs, **loader_args) for domain in datasets_src]
datasets_object_tgt = [data_loader_class(domain, seed, configs=configs, **loader_args) for domain in datasets_tgt]

train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

# Set up logger
log_dir_name = f"scenario_{configs.fan_section}" if is_scenario_mode else f"section_{configs.fan_section}"
full_path_log = os.path.join('Output', 'myMethod', 'log_files_TEST', log_dir_name)
os.makedirs(full_path_log, exist_ok=True)
currtime = time.strftime("%Y%m%d_%H%M%S")
logger = create_logger(os.path.join(full_path_log, f'log_file_TEST_{currtime}.log'))
model.logger = logger

# ==============================================================================
# 3. Execute Testing
# ==============================================================================
if __name__ == '__main__':
    model.eval()

    print("\nStarting evaluation...")
    print(f"Source Domains: {datasets_src}")
    print(f"Target Domains (for testing): {datasets_tgt}")

    # Combine loaders for testing
    all_test_loaders = test_loaders_tgt + test_loaders_src

    if not all_test_loaders:
        print("Error: No test data loaders could be created. Cannot perform evaluation.")
    else:
        # Run the test
        acc_results, auc_results, prec_results, recall_result, f1_results = model.test_model(all_test_loaders)

        # Assuming the first loader in the list is the primary target
        print("\n--- Test Results on Target Domain ---")
        print(f"  Accuracy:  {acc_results[0]:.4f}")
        print(f"  AUC:       {auc_results[0]:.4f}")
        print(f"  Precision: {prec_results[0]:.4f}")
        print(f"  Recall:    {recall_result[0]:.4f}")
        print(f"  F1-Score:  {f1_results[0]:.4f}")

        filename_prefix = f"scenario_{configs.fan_section}" if is_scenario_mode else f"section{configs.fan_section}"
        print("\n--- Filename-style summary ---")
        print(
            f"{filename_prefix}_acc{acc_results[0]:.4f}_auc{auc_results[0]:.4f}_pre{prec_results[0]:.4f}_rec{recall_result[0]:.4f}_f1{f1_results[0]:.4f}")
