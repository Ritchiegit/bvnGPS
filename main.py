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
from data_processing.iPAGE import calculate_delta_and_quantization

from biomarker_select import biomarker_select
from summary_and_train import summary_and_train
if __name__ == "__main__":
    # 1. parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=int(time.time() * 100) % 399, help="")
    parser.add_argument("--classes_num", type=int, default=3, help="")
    parser.add_argument("--relative_sig", type=str, default="test_1")
    parser.add_argument("--dataset", type=str, default="coco_nc2020")  # coconut coco_nc2020

    args = parser.parse_args()
    SEED = args.SEED
    classes_num = args.classes_num
    relative_sig = args.relative_sig
    dataset = args.dataset
    local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(SEED)
    folder_name = f"{relative_sig}_2level_{dataset}_seed{SEED}"
    path_data_prepared = f"data_prepared/{folder_name}/"
    result_2categories_path = f"results/{folder_name}/2categories/"
    result_final_save_path = f"results/{folder_name}/concate/"
    if not os.path.exists(path_data_prepared):
        os.makedirs(path_data_prepared)
    if not os.path.exists(result_2categories_path):
        os.makedirs(result_2categories_path)
    if not os.path.exists(result_final_save_path):
        os.makedirs(result_final_save_path)

    # 2. split the train set/ test set
    gene_GSE, label_GSE_concated = load_data_raw(dataset=dataset)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
    gene_GSE_concated = gene_GSE_concated.T
    gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(gene_GSE_concated, label_GSE_concated, test_size=0.3)

    # 3. 筛选出 pair_index的基因，获得划分阈值的电平
    ## iPAGE 筛选出显著的对
    ### 在训练集上计算出iPAGE的对
    ### 在测试集上同样筛选出与上一行相同的对
    ## 使用LASSO在上面进行训练 获得筛选出来的pair
    pair_index_selected_0, pair_index_exact_expressed_list_final_0 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "control", path_data_prepared, result_2categories_path, local_time=local_time)
    pair_index_selected_1, pair_index_exact_expressed_list_final_1 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "bacteria", path_data_prepared, result_2categories_path, local_time=local_time)
    pair_index_selected_2, pair_index_exact_expressed_list_final_2 = biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test,
                                                                                                                        "virus", path_data_prepared, result_2categories_path, local_time=local_time)
    # 存储ipage、LASSO筛选出的类别个数
    pair_after_ipage_pd = pd.DataFrame({"control": [len(pair_index_exact_expressed_list_final_0)],
                                        "bacteria": [len(pair_index_exact_expressed_list_final_1)],
                                        "virus": [len(pair_index_exact_expressed_list_final_2)]})
    pair_after_ipage_pd.to_csv(result_final_save_path + "pair_after_ipage.csv")
    pair_after_lasso_pd = pd.DataFrame(
        {"control": [pair_index_selected_0.shape[0]], "bacteria": [pair_index_selected_1.shape[0]],
         "virus": [pair_index_selected_2.shape[0]]})
    pair_after_lasso_pd.to_csv(result_final_save_path + "pair_after_lasso.csv")

    def list_with_index(list_in_list, index_in_array):
        new_list = []
        for index_in_array_each in index_in_array:
            new_list.append(list_in_list[index_in_array_each])
        return new_list
    # 4. 根据基因阈值，将绝对表达量转化为相对表达——对应做差，根据阈值划分为4个电平
    pair_after_lasso_0 = list_with_index(pair_index_exact_expressed_list_final_0, pair_index_selected_0)
    pair_after_lasso_1 = list_with_index(pair_index_exact_expressed_list_final_1, pair_index_selected_1)
    pair_after_lasso_2 = list_with_index(pair_index_exact_expressed_list_final_2, pair_index_selected_2)

    # 5. 拼接 训练
    train_data_0 = calculate_delta_and_quantization(pair_after_lasso_0, gene_GSE_concated_train)
    train_data_1 = calculate_delta_and_quantization(pair_after_lasso_1, gene_GSE_concated_train)
    train_data_2 = calculate_delta_and_quantization(pair_after_lasso_2, gene_GSE_concated_train)
    test_data_0 = calculate_delta_and_quantization(pair_after_lasso_0, gene_GSE_concated_test)
    test_data_1 = calculate_delta_and_quantization(pair_after_lasso_1, gene_GSE_concated_test)
    test_data_2 = calculate_delta_and_quantization(pair_after_lasso_2, gene_GSE_concated_test)
    train_data_all = np.concatenate((train_data_0, train_data_1, train_data_2), axis=1)
    test_data_all = np.concatenate((test_data_0, test_data_1, test_data_2), axis=1)

    label_train = get_label_multilabel(label_GSE_concated=label_GSE_concated_train)
    label_test = get_label_multilabel(label_GSE_concated=label_GSE_concated_test)
    train_data_all = train_data_all.astype(np.float32)  # before astype object; after astype float32
    test_data_all = test_data_all.astype(np.float32)  # before astype int16; after astype int32
    label_train = label_train.astype(np.int64)
    label_test = label_test.astype(np.int64)

    summary_and_train(train_data_all, test_data_all, label_train, label_test, result_final_save_path, local_time=local_time, sklearn_random=SEED)
