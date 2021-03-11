import numpy as np
import argparse
import random
import torch
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from data_processing.iPAGE import calculate_delta_and_relative_expression

from biomarker_select import biomarker_select
from summary_and_train import summary_and_train
from utils import load_list_of_tuple, save_list_of_tuple, list_with_index

import itertools
if __name__ == "__main__":
    # 1. parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=int(time.time() * 100) % 399, help="")
    parser.add_argument("--classes_num", type=int, default=3, help="")
    parser.add_argument("--relative_sig", type=str, default="test_1")
    parser.add_argument("--dataset", type=str, default="coco_nc2020")  # coconut coco_nc2020
    parser.add_argument("--dataset_random_state", type=int, default=1, help="")
    parser.add_argument("--test_mode", type=str, default="cohort")  # "cohort", "random"
    parser.add_argument('--test_cohort_index', nargs='+', type=int, default=[1])
    parser.add_argument("--ipage_threshold", type=float, default=1e-16)

    args = parser.parse_args()
    SEED = args.SEED
    classes_num = args.classes_num
    relative_sig = args.relative_sig
    dataset = args.dataset
    dataset_random_state = args.dataset_random_state
    test_mode = args.test_mode  # "cohort45"  # "random"
    local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    test_cohort_index = args.test_cohort_index
    print(test_cohort_index)
    ipage_threshold = args.ipage_threshold
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(SEED)
    if len(test_cohort_index) != 0:
        cohort_str = f"_cohort{test_cohort_index}"
    else:
        cohort_str = ""
    folder_name = f"{relative_sig}_iPAGE_{dataset}_seed{SEED}" \
                  f"_dataRS{dataset_random_state}{cohort_str}_threshold{ipage_threshold}"
    print(folder_name)
    path_data_prepared = f"data_prepared/{folder_name}/"
    result_2categories_path = f"results/{folder_name}/2categories/"
    result_final_save_path = f"results/{folder_name}/concate/"
    result_biomarker = f"results/{folder_name}/biomarker/"
    if not os.path.exists(path_data_prepared):
        os.makedirs(path_data_prepared)
    if not os.path.exists(result_2categories_path):
        os.makedirs(result_2categories_path)
    if not os.path.exists(result_biomarker):
        os.makedirs(result_biomarker)
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
        gene_GSE_concated_train_test_tmp = pd.concat([gene_GSE_concated_train, gene_GSE_concated_test], join="inner", axis=0)
        gene_GSE_concated_train = gene_GSE_concated_train_test_tmp.iloc[:train_lens]
        gene_GSE_concated_test = gene_GSE_concated_train_test_tmp.iloc[train_lens:]
        print("gene_GSE_concated_train.shape", gene_GSE_concated_train.shape)
        print("gene_GSE_concated_test.shape", gene_GSE_concated_test.shape)
    elif test_mode == "random":
        # 2.2random select
        gene_GSE, label_GSE = load_data_raw(dataset=dataset)
        label_GSE_concated = pd.concat(label_GSE, axis=0)
        gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
        gene_GSE_concated = gene_GSE_concated.T
        print(dataset_random_state)
        gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    else:
        print("select unexist test mode")
        input()
        exit(1)


    # 3. 筛选出 pair_index的基因，获得划分阈值的电平
    ## iPAGE 筛选出显著的对
    ### 在训练集上计算出iPAGE的对
    ### 在测试集上同样筛选出与上一行相同的对
    ## 使用LASSO在上面进行训练 获得筛选出来的pair
    pair_index_selected_0, pair_index_exact_expressed_list_final_0 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "control", path_data_prepared, result_2categories_path, local_time=local_time, SEED=SEED, threshold=ipage_threshold)
    pair_index_selected_1, pair_index_exact_expressed_list_final_1 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "bacteria", path_data_prepared, result_2categories_path, local_time=local_time, SEED=SEED, threshold=ipage_threshold)
    pair_index_selected_2, pair_index_exact_expressed_list_final_2 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "virus", path_data_prepared, result_2categories_path, local_time=local_time, SEED=SEED, threshold=ipage_threshold)

    # 4. 总结跳出的pair并存储
    pair_after_lasso_0 = list_with_index(pair_index_exact_expressed_list_final_0, pair_index_selected_0)
    pair_after_lasso_1 = list_with_index(pair_index_exact_expressed_list_final_1, pair_index_selected_1)
    pair_after_lasso_2 = list_with_index(pair_index_exact_expressed_list_final_2, pair_index_selected_2)

    pair_after_lasso_123 = [pair_after_lasso_0, pair_after_lasso_1, pair_after_lasso_2]
    pair_after_lasso = list(set(list(itertools.chain(*pair_after_lasso_123))))
    pair_after_lasso = sorted(pair_after_lasso)
    # 存储ipage、LASSO筛选出的类别个数
    save_list_of_tuple(pair_after_lasso_0, result_biomarker + "pair_after_lasso_0.csv")
    save_list_of_tuple(pair_after_lasso_1, result_biomarker + "pair_after_lasso_1.csv")
    save_list_of_tuple(pair_after_lasso_2, result_biomarker + "pair_after_lasso_2.csv")
    save_list_of_tuple(pair_after_lasso, result_biomarker + "pair_after_lasso.csv")

    num_pair_after_ipage_pd = pd.DataFrame({"control": [len(pair_index_exact_expressed_list_final_0)],"bacteria": [len(pair_index_exact_expressed_list_final_1)],"virus": [len(pair_index_exact_expressed_list_final_2)]})
    num_pair_after_ipage_pd.to_csv(result_biomarker + "num_pair_after_ipage.csv")
    num_pair_after_lasso_pd = pd.DataFrame(
        {"control": [pair_index_selected_0.shape[0]], "bacteria": [pair_index_selected_1.shape[0]],
         "virus": [pair_index_selected_2.shape[0]]})
    num_pair_after_lasso_pd.to_csv(result_biomarker + "num_pair_after_lasso.csv")
    """
    if not os.path.exists(result_final_save_path):
    os.makedirs(result_final_save_path)
    # 5. 获得数据
    train_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_train)
    test_data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated_test)

    label_train = get_label_multilabel(label_GSE_concated=label_GSE_concated_train)
    label_test = get_label_multilabel(label_GSE_concated=label_GSE_concated_test)
    train_data_all = train_data_all.astype(np.float32)  # before astype object; after astype float32
    test_data_all = test_data_all.astype(np.float32)  # before astype int16; after astype int32
    label_train = label_train.astype(np.int64)
    label_test = label_test.astype(np.int64)

    summary_and_train(train_data_all, test_data_all, label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED)
    """