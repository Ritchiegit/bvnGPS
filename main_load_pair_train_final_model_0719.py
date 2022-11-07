import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
import gc
import argparse
from data_processing.iPAGE import calculate_delta_and_relative_expression
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw
from summary_and_train_0719 import summary_and_train
from utils import load_list_of_tuple, list_with_index

parser = argparse.ArgumentParser()
parser.add_argument("--SEED", type=int, default=int(time.time() * 100) % 399, help="")
parser.add_argument("--relative_sig", type=str, default="0723_test")
parser.add_argument("--dataset", type=str, default="coco_nc2020_host")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
parser.add_argument("--dataset_random_state", type=int, default=1, help="")
parser.add_argument("--test_mode", type=str, default="ran07_val_and_exter_val")
parser.add_argument("--pair_path", type=str, default="results/20210325_external2_1_common_gene/20210325_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv")  # "cohort", "random"
parser.add_argument('--test_cohort_GSE', nargs='+', type=str, default=[21802, 57065])
parser.add_argument('--model_str', type=str, default="")
parser.add_argument('--eval_mode', type=str, default="only_three")
args = parser.parse_args()
SEED = args.SEED
relative_sig = args.relative_sig
dataset = args.dataset
dataset_random_state = args.dataset_random_state
test_mode = args.test_mode  # "cohort"  # "random"
pair_path = args.pair_path
test_cohort_GSE = args.test_cohort_GSE
# parameters
local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

print("pair_path", pair_path)
# assert pair_path[-3:] == "-16"
pair_after_lasso_bacteria = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_1.csv")
pair_after_lasso_virus = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_2.csv")
pair_after_lasso_noninfected = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_0.csv")
print(len(pair_after_lasso_bacteria))
print(len(pair_after_lasso_virus))
print(len(pair_after_lasso_noninfected))

if test_mode == "ran07_val_and_exter_val":
    cohort_str = f"_ran07_exter_val{test_cohort_GSE}"
else:
    cohort_str = ""
folder_name = f"{relative_sig}/{relative_sig}_iPAGE_{dataset}_seed{SEED}_dataRS{dataset_random_state}{cohort_str}"
result_final_save_path = f"results/final_model_results/{folder_name}"
if not os.path.exists("results/final_model_results/"):
    os.makedirs("results/final_model_results/")


# 2. split the train set/ test set
# 2.1 cohort
if test_mode == "ran07_val_and_exter_val":
    gene_GSE, label_GSE = load_data_raw(dataset=dataset, external_val_set=test_cohort_GSE)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1).T

    label_GSE_concated = pd.concat(label_GSE, axis=0)

    gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    print("in ran07 exter val")
    # input()
else:
    print("select unexist test mode")
    input()
    exit(1)

label_train = get_label_multilabel(label_GSE_concated=label_GSE_concated_train)
label_test = get_label_multilabel(label_GSE_concated=label_GSE_concated_test)

# calculate the pair
# TODO 2 data selected
train_data_bacteria = calculate_delta_and_relative_expression(pair_after_lasso_bacteria, gene_GSE_concated_train)
train_data_virus = calculate_delta_and_relative_expression(pair_after_lasso_virus, gene_GSE_concated_train)
train_data_noninfected = calculate_delta_and_relative_expression(pair_after_lasso_noninfected, gene_GSE_concated_train)
test_data_bacteria = calculate_delta_and_relative_expression(pair_after_lasso_bacteria, gene_GSE_concated_test)
test_data_virus = calculate_delta_and_relative_expression(pair_after_lasso_virus, gene_GSE_concated_test)
test_data_noninfected = calculate_delta_and_relative_expression(pair_after_lasso_noninfected, gene_GSE_concated_test)
# train_data_all = np.concatenate([train_data_bacteria, train_data_virus, train_data_noninfected], axis=0)
# test_data_all = np.concatenate([test_data_bacteria, test_data_virus, test_data_noninfected], axis=0)
print(train_data_bacteria.shape)
print(train_data_virus.shape)
print(train_data_noninfected.shape)
print(test_data_bacteria.shape)
print(test_data_virus.shape)
print(test_data_noninfected.shape)

# print(train_data_all.shape)
# print(test_data_all.shape)

gc.collect()
# train_data_all = train_data_all.astype(np.float32)  # before astype object; after astype float32
train_data_bacteria = train_data_bacteria.astype(np.float32)  # before astype object; after astype float32
train_data_virus = train_data_virus.astype(np.float32)  # before astype object; after astype float32
train_data_noninfected = train_data_noninfected.astype(np.float32)  # before astype object; after astype float32
# test_data_all = test_data_all.astype(np.float32)  # before astype int16; after astype int32
test_data_bacteria = test_data_bacteria.astype(np.float32)  # before astype int16; after astype int32
test_data_virus = test_data_virus.astype(np.float32)  # before astype int16; after astype int32
test_data_noninfected = test_data_noninfected.astype(np.float32)  # before astype int16; after astype int32
label_train = label_train.astype(np.int64)
label_test = label_test.astype(np.int64)
config = {"bacteria": len(pair_after_lasso_bacteria),
          "virus": len(pair_after_lasso_virus),
          "noninfected": len(pair_after_lasso_noninfected),
          "model_str": args.model_str,
          "eval_mode": args.eval_mode,
}
print(len(pair_after_lasso_bacteria))
print(len(pair_after_lasso_virus))
print(len(pair_after_lasso_noninfected))
# train model

# summary_and_train(
#     [train_data_bacteria, train_data_virus, train_data_noninfected], [test_data_bacteria, test_data_virus, test_data_noninfected],
#     label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED, config=config)

from summary_and_train_0719_sep_train import summary_and_train
summary_and_train(
    [train_data_bacteria, train_data_virus, train_data_noninfected], [test_data_bacteria, test_data_virus, test_data_noninfected],
    label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED, config=config)
