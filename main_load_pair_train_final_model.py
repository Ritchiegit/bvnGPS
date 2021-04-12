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
from summary_and_train import summary_and_train
from utils import load_list_of_tuple, list_with_index

parser = argparse.ArgumentParser()
parser.add_argument("--SEED", type=int, default=int(time.time() * 100) % 399, help="")
parser.add_argument("--relative_sig", type=str, default="test_1")
parser.add_argument("--dataset", type=str, default="coco_nc2020_host")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
parser.add_argument("--dataset_random_state", type=int, default=1, help="")
parser.add_argument("--test_mode", type=str, default="cohort")  # "cohort", "random"
parser.add_argument('--test_cohort_index', nargs='+', type=int, default=[5])
parser.add_argument("--pair_path", type=str, default="results/20210325_external2_1_common_gene/20210325_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv")  # "cohort", "random"
parser.add_argument('--test_cohort_GSE', nargs='+', type=str, default=[21802, 57065])

args = parser.parse_args()
SEED = args.SEED
relative_sig = args.relative_sig
dataset = args.dataset
dataset_random_state = args.dataset_random_state
test_mode = args.test_mode  # "cohort"  # "random"
test_cohort_index = args.test_cohort_index
pair_path = args.pair_path
test_cohort_GSE = args.test_cohort_GSE
# parameters
local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
pair_after_lasso = load_list_of_tuple(pair_path)

if test_mode == "cohort":
    cohort_str = f"_cohort{test_cohort_index}"
elif test_mode == "ran07_val_and_exter_val":
    cohort_str = f"_ran07_exter_val{test_cohort_GSE}"
else:
    cohort_str = ""
folder_name = f"{relative_sig}_iPAGE_{dataset}_seed{SEED}_dataRS{dataset_random_state}{cohort_str}"
result_final_save_path = f"results/final_model_results/{folder_name}"
if not os.path.exists("results/final_model_results/"):
    os.makedirs("results/final_model_results/")


# 2. split the train set/ test set
# 2.1 cohort
if test_mode == "cohort":
    gene_GSE, label_GSE = load_data_raw(dataset=dataset)

    all_index = range(len(gene_GSE))
    # test_index = [4, 5]
    test_index = test_cohort_index

    train_index = list(set(all_index).difference(set(test_index)))
    print("train_index", train_index)
    print("test_index", test_index)
    gene_GSE_train = list_with_index(gene_GSE, train_index)
    gene_GSE_test = list_with_index(gene_GSE, test_index)

    label_GSE_train = list_with_index(label_GSE, train_index)
    label_GSE_test = list_with_index(label_GSE, test_index)

    label_GSE_concated_train = pd.concat(label_GSE_train, axis=0)
    gene_GSE_concated_train = pd.concat(gene_GSE_train, join="inner", axis=1)
    gene_GSE_concated_train = gene_GSE_concated_train.T

    label_GSE_concated_test = pd.concat(label_GSE_test, axis=0)
    gene_GSE_concated_test = pd.concat(gene_GSE_test, join="inner", axis=1)
    gene_GSE_concated_test = gene_GSE_concated_test.T
    train_lens = gene_GSE_concated_train.shape[0]
    test_lens = gene_GSE_concated_test.shape[0]
    print("train_lens", train_lens)
    print("test_lens", test_lens)

    # gene_GSE_concated_train_new, gene_GSE_concated_test_new = select_common_gene_expression_from_train_test(gene_GSE_concated_train, gene_GSE_concated_test)
    # print("gene_GSE_concated_train_new.shape", gene_GSE_concated_train_new.shape)
    # print("gene_GSE_concated_test_new.shape", gene_GSE_concated_test_new.shape)

    # path = f"results/20210311_1_common_gene/20210311_1_data37_marker_iPAGE_coco_nc2020_seed1_dataRS1_cohort{test_cohort_index}_threshold1e-16/biomarker/pair_after_lasso.csv"
    # path = f"results/20210310_leave_one_val/20210310_cohort_val_marker_iPAGE_coco_nc2020_seed1_dataRS1_cohort{test_cohort_index}_threshold1e-16/biomarker/pair_after_lasso.csv"
    # path = f"results/0308_leave_one_val/0308_cohort_val_marker_iPAGE_coco_nc2020_seed2_dataRS2_cohort{test_cohort_index}/biomarker/pair_after_lasso.csv"
    # path = "results/0308_cohort_biomarker_iPAGE_coco_nc2020_seed1_dataRS1_loc20210308_111618/biomarker/pair_after_lasso.csv"  # 45
    # path = "results/20210304_common_gene/0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS1_loc20210304_224937/biomarker/pair_after_lasso.csv"  # 7:3
    pair_after_lasso = load_list_of_tuple(pair_path)

elif test_mode == "random":
    # 2.2random select
    gene_GSE, label_GSE = load_data_raw(dataset=dataset)
    label_GSE_concated = pd.concat(label_GSE, axis=0)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
    gene_GSE_concated = gene_GSE_concated.T
    print(dataset_random_state)
    gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    # load pair selected
    # path = "results/20210228_1_2level_coco_nc2020_seed51/biomarker/pair_after_lasso.csv"
    # path = "results/20210304_common_gene/0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS1_loc20210304_224937/biomarker/pair_after_lasso.csv"
    # path = "results/20210311_1_common_gene/20210311_1_data37_marker_iPAGE_coco_nc2020_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
    # path = "results/20210311_1_common_gene/20210311_1_data37_marker_iPAGE_coco_nc2020_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso_removeHERC2_UBE2C.csv"
    # path = "results/20210323_1_common_gene/20210323_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
    # pair_path = "results/20210325_external2_1_common_gene/20210325_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
    pair_after_lasso = load_list_of_tuple(pair_path)
    print("len(pair_after_lasso)", len(pair_after_lasso))
    # assert len(pair_after_lasso) == 56
elif test_mode == "ran07_val_and_exter_val":
    gene_GSE, label_GSE = load_data_raw(dataset=dataset, external_val_set=test_cohort_GSE)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1).T

    label_GSE_concated = pd.concat(label_GSE, axis=0)
    gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    # pair_after_lasso = load_list_of_tuple(pair_path)
    print("len(pair_after_lasso)", len(pair_after_lasso))

    print("in ran07 exter val")
    # input()
else:
    print("select unexist test mode")
    input()
    exit(1)

label_train = get_label_multilabel(label_GSE_concated=label_GSE_concated_train)
label_test = get_label_multilabel(label_GSE_concated=label_GSE_concated_test)

# calculate the pair
train_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_train)
test_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_test)
gc.collect()
train_data_all = train_data_all.astype(np.float32)  # before astype object; after astype float32
test_data_all = test_data_all.astype(np.float32)  # before astype int16; after astype int32
label_train = label_train.astype(np.int64)
label_test = label_test.astype(np.int64)

# train model
summary_and_train(train_data_all, test_data_all, label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED)
