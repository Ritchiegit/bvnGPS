from utils import load_list_of_tuple, list_with_index
import pandas as pd
import os
import numpy as np
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
import math
import seaborn as sns
os.chdir("..")
# pair_path = "results/20210326_external2_1_common_gene/20210326_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/pair_after_lasso.csv"
label_select = 2
# biomaker_path = "results/20210326_external2_1_common_gene/20210326_external2_1_iPAGE_all_exclude_21802_57065_seed1_dataRS1_threshold1e-16/biomarker/"
# dataset = "all_exclude_21802_57065"

# biomarker_path = "results/20210416_2_e_6_2_common_gene/20210416_2_iPAGE_all_exclude2_6269_1_raw63990_seed1_dataRS1_threshold1e-16/biomarker/"
# dataset = "all_exclude2_6269_1_raw63990"
biomarker_path = "results/20210416_2_e_6_2_common_gene/20210416_2_iPAGE_all_exclude2_6269_1_raw63990_seed1_dataRS1_threshold1e-16/biomarker/"
dataset = "all_exclude2_6269_1_raw63990"
# 11,15,19
# 4 * 3
# 4 * 4
# 4 * 5
pair_path = f"{biomarker_path}/pair_after_lasso_{label_select}.csv"
save_path = f"{biomarker_path}/gene_number_of_reserval{label_select}_training_set.csv"
pair_after_lasso = load_list_of_tuple(pair_path)
print(pair_after_lasso)

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
        # v1 = np.average(gene1_exp)
        # v2 = np.average(gene2_exp)
        pos_1_g_2 = np.sum(gene1_exp > gene2_exp)  # p g
        pos_1_s_2 = np.sum(gene1_exp <= gene2_exp)  # p s
        print("pos_1_g_2, pos_1_s_2", pos_1_g_2, pos_1_s_2)



        gene1_exp = gene_GSE_concated_neg[gene1]
        gene2_exp = gene_GSE_concated_neg[gene2]
        print(gene1_exp.shape)
        print(gene2_exp.shape)
        # v1_1 = np.average(gene1_exp)
        # v2_1 = np.average(gene2_exp)
        neg_1_g_2 = np.sum(gene1_exp > gene2_exp)  # n g
        neg_1_s_2 = np.sum(gene1_exp <= gene2_exp)  # n s
        print("neg_1_g_2, neg_1_s_2", neg_1_g_2, neg_1_s_2)

        # print(gene1_str, gene2_str,  v1, v2, v1_1, v2_1)
        # input()
        save_one = (gene1_str+"_" +gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2)
        final_print.append(save_one)
    final_csv = pd.DataFrame(final_print)
    print(final_csv)
    final_csv.to_csv(save_path, index=False, header=False)
else:
    final_csv = pd.read_csv(save_path, header=None)
    print(final_csv)
import matplotlib
from matplotlib import pyplot as plt
print(final_csv.shape[0],)
col_num = 6
row_num = math.ceil(final_csv.shape[0]/col_num)
fig = plt.figure(figsize=(col_num*2, row_num*2))
# fig, all_ax = plt.subplots(col_num, row_num, figsize=(col_num*2, row_num*2))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i in final_csv.index:
    gene1_gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2 = final_csv.iloc[i]
    print(gene1_gene2_str, pos_1_g_2, pos_1_s_2, neg_1_g_2, neg_1_s_2)
    gene1, gene2 = gene1_gene2_str.split('_', 1)

    trans_mat = np.array([[neg_1_g_2/(neg_1_g_2+neg_1_s_2), neg_1_s_2/(neg_1_g_2+neg_1_s_2)],
                          [pos_1_g_2/(pos_1_g_2+pos_1_s_2), pos_1_s_2/(pos_1_g_2+pos_1_s_2)]], dtype=float)
    # label = ["Patt {}".format(i) for i in range(1, trans_mat.shape[0] + 1)]

    df = pd.DataFrame(trans_mat, index=[1, 2], columns=[1, 2])

    # Plot
    plt.subplot(row_num, col_num, i+1)
    # ax = sns.heatmap(df, xticklabels=df.corr().columns,
    #                  yticklabels    =df.corr().columns, cmap='magma',
    #                  linewidths=6, annot=True)
    # ax = sns.heatmap(df, linewidths=6, cbar=False, xticklabels=["g1>g2", "g1<g2"], yticklabels=["Neg", "Pos"], annot=True, fmt='.20g',
    #                       cmap="Blues")  # Greys	PuRd	RdPu	OrRd	Reds	YlOrRd
    ax = sns.heatmap(df, linewidths=6, cbar=True,
                     xticklabels=["$g_{i}>g_{j}$", "$g_{i} \leq g_{j}$"], yticklabels=["Nega", "Posi"],
                     annot=True, fmt='.0%',
                     vmin=0, vmax=1, cmap="Blues", cbar_ax=cbar_ax)  # Greys	PuRd	RdPu	OrRd	Reds	YlOrRd
    # ax = sns.heatmap(df, linewidths=6, cbar=False, xticklabels=[], yticklabels=[], annot=True, fmt='.20g', cmap="Blues")  # Greys	PuRd	RdPu	OrRd	Reds	YlOrRd
    # ax.set_xticklabels(["1", "2"],rotation='horizontal')
    # plt.axis("off")
    ax.xaxis.tick_top()
    ax.set_yticklabels(["Nega", "Posi"], va="center")
    ax.set_title(f"$\it{gene1}$-$\it{gene2}$")
    # Decorations
    # plt.xticks(fontsize=16, family='Times New Roman')
    # plt.yticks(fontsize=16, family='Times New Roman')
cbar = ax.collections[0].colorbar
cbar.set_ticks([0, .25, .5, .75, 1])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
# fig.tight_layout()
import time
local_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
fig.tight_layout(rect=[0, 0, .9, 1])


plt.savefig(f'{biomarker_path}/pair_count/{dataset}_label{label_select}.png', transparent=True, dpi=800)
plt.show()
