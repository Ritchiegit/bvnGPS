import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split

from data_processing.iPAGE import calculate_delta_and_relative_expression
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw
from summary_and_train import summary_and_train
from utils import load_list_of_tuple

# parameters
dataset = "coco_nc2020"
local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
relative_sig = "20210301"
SEED = 6
folder_name = f"{relative_sig}_iPAGE_{dataset}_seed{SEED}"
result_final_save_path = f"results/final_model_results/{folder_name}"
if not os.path.exists("results/final_model_results/"):
    os.makedirs("results/final_model_results/")

# load pair selected
path = "results/20210228_1_2level_coco_nc2020_seed51/biomarker/pair_after_lasso.csv"
pair_after_lasso = load_list_of_tuple(path)

# load data
gene_GSE, label_GSE_concated = load_data_raw(dataset=dataset)
gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
gene_GSE_concated = gene_GSE_concated.T
gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(
    gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=SEED)
label_train = get_label_multilabel(label_GSE_concated=label_GSE_concated_train)
label_test = get_label_multilabel(label_GSE_concated=label_GSE_concated_test)

# calculate the pair
train_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_train)
test_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_test)

train_data_all = train_data_all.astype(np.float32)  # before astype object; after astype float32
test_data_all = test_data_all.astype(np.float32)  # before astype int16; after astype int32
label_train = label_train.astype(np.int64)
label_test = label_test.astype(np.int64)

# train model
summary_and_train(train_data_all, test_data_all, label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED)
