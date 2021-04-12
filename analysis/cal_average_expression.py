from utils import load_list_of_tuple, list_with_index
import pandas as pd
import os
import numpy as np
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
os.chdir("..")
# pair_path = "results/20210326_external2_1_common_gene/20210326_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
label_select = 0
# label_select = 1
# label_select = 2

pair_path = f"results/20210326_external2_1_common_gene/20210326_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso_{label_select}.csv"
pair_after_lasso = load_list_of_tuple(pair_path)
print(pair_after_lasso)


from load_data.load_data_raw import load_data_raw

# gene_GSE, label_GSE = load_data_raw("coco_nc2020_host")
# label_GSE_concated = pd.concat(label_GSE, axis=0)
# gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
# gene_GSE_concated = gene_GSE_concated.T
# path_model = "results/final_model_results/20210326_external2_1_model_selected_0328/pred_result/2/all_exclude_21802_570650.7/"
dataset = "all_exclude_21802_57065"
type_part_dataset = "0.7"
dataset_random_state = 1
gene_GSE, label_GSE = load_data_raw(dataset=dataset)
label_GSE_concated = pd.concat(label_GSE, axis=0)
gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
gene_GSE_concated = gene_GSE_concated.T
if type_part_dataset == "0.7":
    gene_GSE_concated, _, label_GSE_concated, _ = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    # print("type 0.7")
    # print(gene_GSE_concated.shape)
elif type_part_dataset == "0.3":
    _, gene_GSE_concated, _, label_GSE_concated = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    print("type 0.3")
    print(gene_GSE_concated.shape)
else:
    assert type_part_dataset is None

label_raw = get_label_multilabel(label_GSE_concated)
label = label_raw == label_select  # 正标签
label_neg = label_raw != label_select
print(label)
print(label_neg)
print(gene_GSE_concated.shape)
final_print = []
print("label", sum(label))
print("label_neg", sum(label_neg))
gene_GSE_concated_pos = gene_GSE_concated.iloc[label]
gene_GSE_concated_neg = gene_GSE_concated.iloc[label_neg]

print(gene_GSE_concated_pos.shape)
print(gene_GSE_concated_neg.shape)
for pair in pair_after_lasso:
    gene1, gene2 = pair
    gene1_str = gene1[1]
    gene2_str = gene2[1]

    gene1_exp = gene_GSE_concated_pos[gene1]
    gene2_exp = gene_GSE_concated_pos[gene2]
    print(gene1_exp.shape)
    print(gene2_exp.shape)
    v1 = np.average(gene1_exp)
    v2 = np.average(gene2_exp)
    v1_log2 = np.log2(v1)
    v2_log2 = np.log2(v2)

    gene1_exp = gene_GSE_concated_neg[gene1]
    gene2_exp = gene_GSE_concated_neg[gene2]
    print(gene1_exp.shape)
    print(gene2_exp.shape)
    v1_1 = np.average(gene1_exp)
    v2_1 = np.average(gene2_exp)
    v1_1_log2 = np.log2(v1_1)
    v2_1_log2 = np.log2(v2_1)

    print(gene1_str, gene2_str,  v1, v2, v1_1, v2_1)
    print(gene1_str, gene2_str,  v1_log2, v2_log2, v1_1_log2, v2_1_log2)
    # input()
    # save_one = (gene1_str+"_" +gene2_str, v1, v2, v1_1, v2_1)
    save_one = (gene1_str+"_" +gene2_str, v1_log2, v2_log2, v1_1_log2, v2_1_log2)
    final_print.append(save_one)
final_csv = pd.DataFrame(final_print)
final_csv.to_csv(f"analysis/gene_expression_training_set/log2_gene_expression{label_select}_training_set.csv", index=False, header=False)