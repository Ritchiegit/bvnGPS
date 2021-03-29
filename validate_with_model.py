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

if __name__ == "__main__":
    # dataset = "only_21802_57065"
    # dataset = "COVID19"
    # model_folder_name = "20210323_1_data37_SVM/" + "20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_model/"
    # model_path = f"results/final_model_results/{model_folder_name}"
    # result_folder_name = "20210323_1_data37_SVM/" + "validation/" + "20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_validation.csv"
    # result_final_save_path = f"results/final_model_results/{result_folder_name}"
    #
    # if not os.path.exists("results/final_model_results/20210323_1_data37_SVM/validation/"):
    #     os.makedirs("results/final_model_results/20210323_1_data37_SVM/validation/")

    # total_folder = "20210323_external2_model_iPAGE_all_exclude/"
    # one_model = "20210323_external2_model_iPAGE_all_exclude_21802_57065_seed10_dataRS1_model"
    # total_folder = "20210324_external2_1_model_iPAGE_all_exclude/"
    # one_model = "20210324_external2_1_model_iPAGE_all_exclude_21802_57065_seed111_dataRS1_model"
    # total_folder = "20210324_test_save_iPAGE_all/"
    # one_model = "20210324_test_save_iPAGE_all_exclude_21802_57065_seed262_dataRS1_model"
    # total_folder = "20210324_external2_1_model_iPAGE_all_exclude/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="coco_nc2020")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
    parser.add_argument("--total_folder", type=str,
                        default="20210324_external2_1_model_iPAGE_all_exclude/")  # coconut coco_nc2020 GSE6269 all_exclude_21802_57065 only_21802_57065
    parser.add_argument("--pair_path", type=str,
                        default="results/20210325_external2_1_common_gene/20210325_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv")  # "cohort", "random"
    args = parser.parse_args()
    pair_path = args.pair_path
    dataset = args.dataset
    total_folder = args.total_folder

    model_list = os.listdir(f"results/final_model_results/{total_folder}")
    for one_model in model_list:
        model_folder_name = total_folder + f"{one_model}/"
        model_path = f"results/final_model_results/{model_folder_name}"
        result_folder_name = total_folder + f"on{dataset}_validation/" + f"{one_model}.csv"
        result_final_save_path = f"results/final_model_results/{result_folder_name}"

        if not os.path.exists(f"results/final_model_results/{total_folder}/on{dataset}_validation/"):
            os.makedirs(f"results/final_model_results/{total_folder}/on{dataset}_validation/")

        dataset_random_state = 1
        gene_GSE, label_GSE = load_data_raw(dataset=dataset)
        label_GSE_concated = pd.concat(label_GSE, axis=0)
        gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
        gene_GSE_concated = gene_GSE_concated.T
        print(gene_GSE_concated.shape)
        print(label_GSE_concated.shape)
        print(dataset_random_state)
        # pair_path = "results/20210311_1_common_gene/20210311_1_data37_marker_iPAGE_coco_nc2020_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso_removeHERC2_UBE2C.csv"
        # pair_path = "results/20210323_1_common_gene/20210323_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
        # pair_path = "results/20210323_1_common_gene/20210323_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
        # pair_path = "results/20210324_external2_1_common_gene/20210324_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS0_threshold1e-16/biomarker/pair_after_lasso.csv"
        pair_after_lasso = load_list_of_tuple(pair_path)
        # assert len(pair_after_lasso) == 49
        # assert len(pair_after_lasso) == 56
        # assert len(pair_after_lasso) == 51
        print(len(pair_after_lasso))

        label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
        data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated)

        gc.collect()
        data_all = data_all.astype(np.float32)  # before astype object; after astype float32
        label = label.astype(np.int64)

        # model_path = "final_model_results/20210323_1_data37_SVM/20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_model/"
        summary_and_test(data_all, label, model_path, result_final_save_path, local_time=0, sklearn_random=1)
