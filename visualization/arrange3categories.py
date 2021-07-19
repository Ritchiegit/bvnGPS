import pandas as pd
import numpy as np
from load_data.load_data_raw import load_data_raw
from sklearn.model_selection import train_test_split
from data_processing.process_data_label import get_label_multilabel
import os
from matplotlib import pyplot as plt

os.chdir("..")
dataset = "only_21802_57065"
pred_path = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/only_21802_57065None/threeNeuralNetworkMSE_optAdam_h1024_lr0.0005_val0.0589126533185932_epoch10.pth.csv"
type_part_dataset = None

# pred_path = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/all_exclude2_6269_1_raw639900.3/threeNeuralNetworkMSE_optAdam_h1024_lr0.0005_val0.0589126533185932_epoch10.pth.csv"
# dataset = "all_exclude2_6269_1_raw63990"
# type_part_dataset = "0.3"

# pred_path = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/all_exclude2_6269_1_raw639900.7/threeNeuralNetworkMSE_optAdam_h1024_lr0.0005_val0.0589126533185932_epoch10.pth.csv"
# dataset = "all_exclude2_6269_1_raw63990"
# type_part_dataset = "0.7"
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

label = get_label_multilabel(label_GSE_concated=label_GSE_concated)

pred_3 = pd.read_csv(pred_path, header=None)
pred_3 = pred_3.values

label_0_index = label == 0
pred_3_0 = pred_3[label_0_index, :]

label_1_index = label == 1
pred_3_1 = pred_3[label_1_index, :]


label_2_index = label == 2
pred_3_2 = pred_3[label_2_index, :]

len_0 = pred_3_0.shape[0]
len_1 = pred_3_1.shape[0]
len_2 = pred_3_2.shape[0]

print(pred_3_0.shape)
print(pred_3_1.shape)
print(pred_3_2.shape)

pred_concated = np.concatenate((pred_3_1, pred_3_2, pred_3_0), axis=0)
print(pred_concated.shape)

plt.figure(figsize=(13, 4), dpi=300)

plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 1], marker="^", c='',edgecolors='chocolate', label='Bacterial')
plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 2], marker="s", c='',edgecolors='steelblue', label='Viral')
plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 0], marker="o", c='',edgecolors='gold', label='Noninfected')
plt.vlines(len_1, 0, 1, color="k", linestyles="--")
plt.vlines(len_1+len_2, 0, 1, color="k", linestyles="--")
plt.xticks([len_1/2,len_1+len_2/2, len_1 + len_2 + len_0/2],['Bacterial', 'Viral', 'Noninfected'], fontsize=15)
plt.xlim((0, len_0+len_1+len_2))
plt.ylim((0, 1))
plt.ylabel('Probability', fontsize=15)
plt.grid()
plt.legend(loc="right", fontsize=15)
plt.tight_layout()

plt.savefig(f'{pred_path[:-4]}_probability_of_3class_0513.png')
