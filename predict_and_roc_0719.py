import pandas as pd
from load_data.load_data_raw import load_data_raw
from sklearn.model_selection import train_test_split
from utils import load_list_of_tuple, list_with_index, check_front_of_name, check_end_of_name
from data_processing.process_data_label import get_label_multilabel
from data_processing.iPAGE import calculate_delta_and_relative_expression
import gc
import numpy as np
import os
import argparse
from test_model import test_sklearn, test_pytorch, test_lda_RF
from test_model_0719 import test_pytorch_three_input_model, test_pytorch_three_input_model_subnet


def val_with_one_dataset(data, label, model_path, model_type, model_name, result_final_save_file):
    AUC_0, AUC_1, AUC_2 = -1, -1, -1
    if model_type == "three_input_model":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_pytorch_three_input_model(model_path, data, label, result_final_save_file, model_name)
    elif model_type == "three_input_model_subnet":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_pytorch_three_input_model_subnet(model_path, data, label, result_final_save_file, model_name)
    elif model_type == "neural network":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_pytorch(model_path, data, label, result_final_save_file, model_name)
    elif model_type == "lda rf":
        lda, rf = model_path
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_lda_RF(lda, rf, data, label, result_final_save_file, model_name="RandomForest")
    elif model_type == "sklearn":
        AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = test_sklearn(model_path, data, label, result_final_save_file, model_name)
    return AUC_0, AUC_1, AUC_2, y_pred, y_pred_3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco_nc2020")
    parser.add_argument('--dataset_list', nargs='+', type=str, default=[])
    parser.add_argument("--model_total_folder_name", type=str, default="test/")
    parser.add_argument("--pair_path", type=str, default="pair_after_lasso.csv")
    parser.add_argument("--type_part_dataset", type=str, default=None)
    parser.add_argument('--exclude_cohort_GSE', nargs='+', type=str, default=[])
    parser.add_argument("--test_mode", type=str, default="exclude_exter_val")
    parser.add_argument("--dataset_random_state", type=int, default=1, help="")

    args = parser.parse_args()
    pair_path = args.pair_path
    dataset = args.dataset
    model_total_folder_name = args.model_total_folder_name
    type_part_dataset = args.type_part_dataset
    exclude_cohort_GSE = args.exclude_cohort_GSE
    test_mode = args.test_mode
    dataset_list = args.dataset_list
    if len(dataset_list) == 0:
        pass
    else:
        dataset = dataset_list

    # pair_after_lasso = load_list_of_tuple(pair_path)
    print(pair_path)
    # assert pair_path[-3:] == "-16"
    pair_after_lasso_bacteria = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_1.csv")
    pair_after_lasso_virus = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_2.csv")
    pair_after_lasso_noninfected = load_list_of_tuple(pair_path + "/biomarker/pair_after_lasso_0.csv")
    print(len(pair_after_lasso_bacteria))
    print(len(pair_after_lasso_virus))
    print(len(pair_after_lasso_noninfected))

    # 处理数据
    dataset_random_state = 1


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

    # data = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated)
    # gc.collect()
    # data = data.astype(np.float32)  # before astype object; after astype float32


    label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
    # data = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated)
    data_bacteria = calculate_delta_and_relative_expression(pair_after_lasso_bacteria, gene_GSE_concated)
    data_virus = calculate_delta_and_relative_expression(pair_after_lasso_virus, gene_GSE_concated)
    data_noninfected = calculate_delta_and_relative_expression(pair_after_lasso_noninfected,
                                                                     gene_GSE_concated)

    gc.collect()
    data_bacteria = data_bacteria.astype(np.float32)  # before astype object; after astype float32
    data_virus = data_virus.astype(np.float32)  # before astype object; after astype float32
    data_noninfected = data_noninfected.astype(np.float32)  # before astype object; after astype float32
    label = label.astype(np.int64)
    print("bacteria", data_bacteria.shape)
    print("virus", data_virus.shape)
    print("noninfected", data_noninfected.shape)
    data = np.concatenate([data_bacteria, data_virus, data_noninfected], axis=1)


    # 数据 模型（批量）
    # 7 3 external
    base_path = "results/final_model_results/"
    model_total_folder_name_path = f"{base_path}{model_total_folder_name}/"  # results/20210325_external_57065_2_iPAGE_all_exclude_57065
    model_one_folder_name_list = os.listdir(model_total_folder_name_path)  # 很多 20210325_external_57065_2_iPAGE_all_exclude_57065_seed111_dataRS1_model
    result_final_save_path = model_total_folder_name_path + f"predict_on{dataset}{type_part_dataset}_validation/"

    print("model_total_folder_name_path", model_total_folder_name_path)
    print("model_one_folder_name_list", model_one_folder_name_list)
    print("result_final_save_path", result_final_save_path)
    if not os.path.exists(result_final_save_path):
        os.makedirs(result_final_save_path)
    for model_one_folder_name in model_one_folder_name_list:
        # if not check_front_of_name(model_one_folder_name, "2"):
        print(model_one_folder_name)
        if not check_front_of_name(model_one_folder_name, "2"):
            continue
        if not check_end_of_name(model_one_folder_name, "model"):
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
            print("model_name", model_name)
            if "three_input_model_concatenate_subnet_train" in model_name:
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset([data_bacteria, data_virus, data_noninfected], label, model_path,
                                                                             "three_input_model_subnet", model_name, result_final_save_file)

            elif "three_input_model_add" in model_name:
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset([data_bacteria, data_virus, data_noninfected], label, model_path,
                                                                             "three_input_model", model_name, result_final_save_file)
            elif "three_input_model_concatenate" in model_name:
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset([data_bacteria, data_virus, data_noninfected], label, model_path,
                    "three_input_model", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "CART"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "sklearn", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "SVM"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "sklearn", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "NN"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "neural network", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "MFCN"):
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "neural network", model_name, result_final_save_file)
            elif check_front_of_name(model_name, "lda"):
                model_path = (model_folder_name+"lda.model_pickle", model_folder_name+"RandomForest.model_pickle")
                AUC_0, AUC_1, AUC_2, y_pred, y_pred_3 = val_with_one_dataset(data, label, model_path, "lda rf", model_name, result_final_save_file)
            else:
                assert check_front_of_name(model_name, "RandomForest"), print(model_name)
                continue
            # 保存预测值
            pred_file = model_pred_path + "/three" + model_name + ".csv"
            y_pred_3_pd = pd.DataFrame(y_pred_3)
            y_pred_3_pd.to_csv(pred_file, index=False, header=False)