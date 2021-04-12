import pandas as pd
from load_data.load_data_raw import load_data_raw
from sklearn.model_selection import train_test_split
from utils import load_list_of_tuple, list_with_index
from data_processing.process_data_label import get_label_multilabel
from data_processing.iPAGE import calculate_delta_and_relative_expression
import gc
import numpy as np
from summary_and_test import summary_and_test
import os
import argparse
from summary_and_test import test_sklearn, test_pytorch, test_lda_RF

def val_with_one_dataset(data, label, model_path, model_type, model_name, result_final_save_file):
    AUC_0, AUC_1, AUC_2 = -1, -1, -1
    if model_type == "neural network":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_pytorch(model_path, data, label, result_final_save_file, model_name)
    elif model_type == "lda rf":
        lda, rf = model_path
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_lda_RF(lda, rf, data, label, result_final_save_file, model_name="RandomForest")
    elif model_type == "sklearn":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_sklearn(model_path, data, label, result_final_save_file, model_name)
    return AUC_0, AUC_1, AUC_2, y_pred, y_pred_3

# 结果应该存在所属文件夹旁边 的 valdatasetname 文件夹中


def check_front_of_name(name, front):
    return name[:len(front)] == front


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="coco_nc2020")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
    parser.add_argument('--dataset_list', nargs='+', type=str, default=[])

    parser.add_argument("--model_total_folder_name", type=str,
                        default="20210324_external2_1_model_iPAGE_all_exclude/")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
    parser.add_argument("--pair_path", type=str,
                        default="results/20210325_external2_1_common_gene/20210325_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv")  # "cohort", "random"
    parser.add_argument("--type_part_dataset", type=str, default=None)
    parser.add_argument('--exclude_cohort_GSE', nargs='+', type=str, default=[])
    parser.add_argument("--test_mode", type=str, default="cohort")  # "cohort", "random"
    parser.add_argument("--dataset_random_state", type=int, default=1, help="")

    args = parser.parse_args()
    pair_path = args.pair_path
    dataset = args.dataset
    model_total_folder_name = args.model_total_folder_name
    type_part_dataset = args.type_part_dataset
    exclude_cohort_GSE = args.exclude_cohort_GSE
    test_mode = args.test_mode  # "cohort"  # "random"
    dataset_list = args.dataset_list
    if len(dataset_list) == 0:
        pass
    else:
        dataset = dataset_list

    pair_after_lasso = load_list_of_tuple(pair_path)

    # 处理数据
    dataset_random_state = 1
    # gene_GSE, label_GSE = load_data_raw(dataset=dataset)
    # label_GSE_concated = pd.concat(label_GSE, axis=0)
    # gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
    # gene_GSE_concated = gene_GSE_concated.T


    if test_mode == "exclude_exter_val":
        gene_GSE, label_GSE = load_data_raw(dataset=dataset, external_val_set=exclude_cohort_GSE)
        gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1).T
        label_GSE_concated = pd.concat(label_GSE, axis=0)
    else:
        exit(1)
    if type_part_dataset == "0.7":
        gene_GSE_concated, _, label_GSE_concated, _ = train_test_split(
            gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
        print("type 0.7")
        print(gene_GSE_concated.shape)
    elif type_part_dataset == "0.3":
        _, gene_GSE_concated, _, label_GSE_concated = train_test_split(
            gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
        print("type 0.3")
        print(gene_GSE_concated.shape)
    else:
        assert type_part_dataset is None

    label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
    data = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated)

    # 这里和 pickle 比较一下是否完全相同
    # 没法比较，直接比较 对比 validation的结果吧
    gc.collect()
    data = data.astype(np.float32)  # before astype object; after astype float32
    label = label.astype(np.int64)
    print(len(pair_after_lasso))


    # 数据 模型（批量）
    # 7 3 external
    base_path = "results/final_model_results/"
    model_total_folder_name_path = f"{base_path}{model_total_folder_name}/"  # results/20210325_external_57065_2_iPAGE_all_exclude_57065
    model_one_folder_name_list = os.listdir(model_total_folder_name_path)  # 很多 20210325_external_57065_2_iPAGE_all_exclude_57065_seed111_dataRS1_model
    result_final_save_path = model_total_folder_name_path + f"predict_on{dataset}{type_part_dataset}_validation/"
    if not os.path.exists(result_final_save_path):
        os.makedirs(result_final_save_path)
    for model_one_folder_name in model_one_folder_name_list:
        # if not check_front_of_name(model_one_folder_name, "2"):
        if not check_front_of_name(model_one_folder_name, "2"):
            continue
        model_folder_name = model_total_folder_name_path + model_one_folder_name + "/"  # results/20210325_external_57065_2_iPAGE_all_exclude_57065/20210325_external_57065_2_iPAGE_all_exclude_57065_seed111_dataRS1_model
        model_list = os.listdir(model_folder_name)  #
        result_final_save_file = result_final_save_path + f"{model_one_folder_name}.csv"
        f = open(result_final_save_file, "w")
        f.close()
        model_list.sort()
        model_pred_path = model_total_folder_name_path + f"/pred_result/{model_one_folder_name}/{dataset}{type_part_dataset}/"
        if not os.path.exists(model_pred_path):
            os.makedirs(model_pred_path)
        for model_name in model_list:
            model_path = model_folder_name + model_name
            if check_front_of_name(model_name, "CART"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "sklearn", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "SVM"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "sklearn", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "NeuralNetwork"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "neural network", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "MFCN"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "neural network", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "lda"):
                model_path = (model_folder_name+"lda.model_pickle", model_folder_name+"RandomForest.model_pickle")
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "lda rf", model_name, result_final_save_file)
            else:
                assert check_front_of_name(model_name, "RandomForest"), print(model_name)
                continue
                # input()
            # pred_file = model_pred_path + "/one" + model_name + ".csv"
            # print(y_pred)
            # y_pred_pd = pd.DataFrame(y_pred)
            # y_pred_pd.to_csv(pred_file, index=False, header=False)

            pred_file = model_pred_path + "/three" + model_name + ".csv"
            # print(y_pred_3)
            y_pred_3_pd = pd.DataFrame(y_pred_3)
            y_pred_3_pd.to_csv(pred_file, index=False, header=False)
        # model_path = "final_model_results/20210323_1_data37_SVM/20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_model/"
        # summary_and_test(data, label, model_path, result_final_save_file, local_time=0, sklearn_random=1)
