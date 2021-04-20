import pandas as pd
import numpy as np
from load_data.load_data_raw import load_data_raw
from sklearn.model_selection import train_test_split
from data_processing.process_data_label import get_label_multilabel
import os
os.chdir("..")
# pred_path = "../label_for_validation/pred_only_21802_57065.csv"
# label_path = "../label_for_validation/label_only_21802_57065.csv"

#
# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.3.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.3.csv"

# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.7.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.7.csv"


pred_path = "results/final_model_results/20210416_2/pred_result/output_selected/only_21802_57065None/threeNeuralNetworkMSE_optAdagrad_h768_lr0.01_val0.06611447938731019_epoch16.pth.csv"
dataset = "only_21802_57065"
type_part_dataset = None

pred_path = "results/final_model_results/20210416_2/pred_result/output_selected/all_exclude2_6269_1_raw639900.3/threeNeuralNetworkMSE_optAdagrad_h768_lr0.01_val0.06611447938731019_epoch16.pth.csv"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"

pred_path = "results/final_model_results/20210416_2/pred_result/output_selected/all_exclude2_6269_1_raw639900.7/threeNeuralNetworkMSE_optAdagrad_h768_lr0.01_val0.06611447938731019_epoch16.pth.csv"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.7"
# path_model = "results/final_model_results/20210416_2/pred_result/output_selected/all_exclude2_6269_1_raw639900.3/threeNeuralNetworkMSE_optAdagrad_h768_lr0.01_val0.06611447938731019_epoch16.pth.csv"
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
# pred_path_list = glob.glob(path_model + "*")

pred_3 = pd.read_csv(pred_path, header=None)
pred_3 = pred_3.values
# label = pd.read_csv(label_path, header=None)
# label = label.values
# label = label.flatten()
# print(label)
# print(pred_3)


# label = label.values.flatten()
# label_one = (label == i) * 1
# pred_one = pred_3[:, i]
# pred_one = pred_one.flatten()

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

from matplotlib import pyplot as plt
plt.figure(figsize=(13, 4), dpi=300)

# plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 0], marker="o", c='',edgecolors='k')
# plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 1], marker="^", c='',edgecolors='DarkBlue')
# plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 2], marker="s", c='',edgecolors='SlateGray')
plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 1], marker="^", c='',edgecolors='chocolate', label='Bacterial')
plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 2], marker="s", c='',edgecolors='steelblue', label='Viral')
plt.scatter(np.arange(pred_concated.shape[0]), pred_concated[:, 0], marker="o", c='',edgecolors='gold', label='Noninfected')
plt.vlines(len_1, 0, 1, color="k", linestyles="--")
plt.vlines(len_1+len_2, 0, 1, color="k", linestyles="--")
plt.xticks([len_1/2,len_1+len_2/2, len_1 + len_2 + len_0/2],['Bacterial', 'Viral', 'Noninfected'])
plt.xlim((0, len_0+len_1+len_2))
plt.ylim((0, 1))
# plt.xlabel('Clinically phenotype')
plt.ylabel('Probability')  # 可以使用中文，但需要导入一些库即字体
plt.grid()
plt.legend(loc="right")
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width , box.height* 0.8])
plt.tight_layout()

plt.savefig(f'{pred_path[:-4]}_probability_of_3class.png')

# plt.show()
