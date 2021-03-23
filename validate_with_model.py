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
if __name__ == "__main__":
    dataset = "GSE6269"


    model_folder_name = "20210323_1_data37_SVM/" + "20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_model/"
    model_path = f"results/final_model_results/{model_folder_name}"
    result_folder_name = "20210323_1_data37_SVM/" + "validation/" + "20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_validation.csv"
    result_final_save_path = f"results/final_model_results/{result_folder_name}"

    if not os.path.exists("results/final_model_results/20210323_1_data37_SVM/validation/"):
        os.makedirs("results/final_model_results/20210323_1_data37_SVM/validation/")

    dataset_random_state = 1
    gene_GSE, label_GSE = load_data_raw(dataset=dataset)
    label_GSE_concated = pd.concat(label_GSE, axis=0)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
    gene_GSE_concated = gene_GSE_concated.T
    print(gene_GSE_concated.shape)
    print(label_GSE_concated.shape)
    print(dataset_random_state)
    pair_path = "results/20210311_1_common_gene/20210311_1_data37_marker_iPAGE_coco_nc2020_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso_removeHERC2_UBE2C.csv"
    pair_after_lasso = load_list_of_tuple(pair_path)
    assert len(pair_after_lasso) == 49
    print(pair_after_lasso)

    label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
    data_all = calculate_delta_and_relative_expression(pair_after_lasso, gene_GSE_concated)

    gc.collect()
    data_all = data_all.astype(np.float32)  # before astype object; after astype float32
    label = label.astype(np.int64)

    # model_path = "final_model_results/20210323_1_data37_SVM/20210323_1_data37_SVM_iPAGE_coco_nc2020_seed102_dataRS1_model/"
    summary_and_test(data_all, label, model_path, result_final_save_path, local_time=0, sklearn_random=1)
