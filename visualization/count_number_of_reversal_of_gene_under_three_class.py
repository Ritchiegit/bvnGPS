from utils import load_list_of_tuple
import pandas as pd
import os
import numpy as np
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
from matplotlib import pyplot as plt
os.chdir("..")
biomarker_path = "results/20210416_2_e_6_2_common_gene/20210416_2_iPAGE_all_exclude2_6269_1_raw63990_seed1_dataRS1_threshold1e-16/biomarker/"
dataset = "all_exclude2_6269_1_raw63990"
pair_num_list = []
last_class_end = 0
col_num = 6
row_num_sum = 0
for label_select in range(3):
    pair_path = f"{biomarker_path}/pair_after_lasso_{label_select}.csv"
    pair_after_lasso = load_list_of_tuple(pair_path)
    pair_num_list.append(len(pair_after_lasso))
    row_num_sum += math.ceil(len(pair_after_lasso) / col_num)
fig = plt.figure(figsize=(col_num*2, row_num_sum*2))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for label_select in [1, 2, 0]:
    pair_path = f"{biomarker_path}/pair_after_lasso_{label_select}.csv"
    save_path = f"{biomarker_path}/gene_number_of_reserval{label_select}_training_set.csv"
    pair_after_lasso = load_list_of_tuple(pair_path)
    if not os.path.exists(biomarker_path+"pair_count/"):
        os.makedirs(biomarker_path+"pair_count/")
    from load_data.load_data_raw import load_data_raw
    if not os.path.exists(save_path):
        type_part_dataset = "0.7"
        dataset_random_state = 1
        gene_GSE, label_GSE = load_data_raw(dataset=dataset)
        label_GSE_concated = pd.concat(label_GSE, axis=0)
        gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
        gene_GSE_concated = gene_GSE_concated.T
        if type_part_dataset == "0.7":
            gene_GSE_concated, _, label_GSE_concated, _ = train_test_split(
                gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
        elif type_part_dataset == "0.3":
            _, gene_GSE_concated, _, label_GSE_concated = train_test_split(
                gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
        else:
            assert type_part_dataset is None

        label_raw = get_label_multilabel(label_GSE_concated)
        label = label_raw == label_select  # 正标签
        label_neg = label_raw != label_select
        final_print = []
        gene_GSE_concated_pos = gene_GSE_concated.iloc[label]
        gene_GSE_concated_neg = gene_GSE_concated.iloc[label_neg]

        for pair in pair_after_lasso:
            gene1, gene2 = pair
            gene1_str = gene1[1]
            gene2_str = gene2[1]
            gene1_exp = gene_GSE_concated_pos[gene1]
            gene2_exp = gene_GSE_concated_pos[gene2]
            pos_1_g_2 = np.sum(gene1_exp > gene2_exp)  # p g
            pos_1_s_2 = np.sum(gene1_exp <= gene2_exp)  # p s

            gene1_exp = gene_GSE_concated_neg[gene1]
            gene2_exp = gene_GSE_concated_neg[gene2]
            neg_1_g_2 = np.sum(gene1_exp > gene2_exp)  # n g
            neg_1_s_2 = np.sum(gene1_exp <= gene2_exp)  # n s
            save_one = (gene1_str+"_" +gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2)
            final_print.append(save_one)
        final_csv = pd.DataFrame(final_print)
        final_csv.to_csv(save_path, index=False, header=False)
    else:
        final_csv = pd.read_csv(save_path, header=None)
    col_num = 6
    row_num = math.ceil(final_csv.shape[0]/col_num)
    for i in final_csv.index:
        gene1_gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2 = final_csv.iloc[i]
        print(gene1_gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2)
        gene1, gene2 = gene1_gene2_str.split('_', 1)
        trans_mat = np.array([[neg_1_g_2/(neg_1_g_2+neg_1_s_2), neg_1_s_2/(neg_1_g_2+neg_1_s_2)],
                              [pos_1_g_2/(pos_1_g_2+pos_1_s_2), pos_1_s_2/(pos_1_g_2+pos_1_s_2)]], dtype=float)
        df = pd.DataFrame(trans_mat, index=[1, 2], columns=[1, 2])
        plt.subplot(row_num_sum, col_num, last_class_end+i+1)
        ax = sns.heatmap(df, linewidths=6, cbar=True,
                         xticklabels=["$g_{i}>g_{j}$", "$g_{i} \leq g_{j}$"], yticklabels=["Nega", "Posi"],
                         annot=True, fmt='.0%',
                         vmin=0, vmax=1, cmap="Blues", cbar_ax=cbar_ax)  # Greys	PuRd	RdPu	OrRd	Reds	YlOrRd
        ax.xaxis.tick_top()
        ax.set_yticklabels(["Nega", "Posi"], va="center")
        ax.set_title(f"$\it{gene1}$-$\it{gene2}$")
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    fig.tight_layout(rect=[0, 0, .9, 1])
    last_class_end += col_num * row_num
plt.savefig(f'{biomarker_path}/pair_count/{dataset}_summary_dpi400.png', transparent=True, dpi=500)
plt.show()
