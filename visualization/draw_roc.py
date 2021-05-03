import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
os.chdir("..")

from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
path_model = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/only_21802_57065None/"
dataset = "only_21802_57065"
type_part_dataset = None
path_model = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/all_exclude2_6269_1_raw639900.3/"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"
path_model = "results/final_model_results/20210416_3_smx/pred_result/20210416_3_smx_iPAGE_all_exclude2_6269_1_raw63990_seed122_dataRS1_ran07_exter_val[21802, 57065]_model/all_exclude2_6269_1_raw639900.7/"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.7"
pred_file_list = os.listdir(path_model)
print("pred_file_list", len(pred_file_list))



path_png = f'{path_model}/png/'
if not os.path.exists(path_png):
    os.makedirs(path_png)

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

label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
# pred_path_list = glob.glob(path_model + "*")
for pred_file in pred_file_list:
    pred_path = path_model + "/" + pred_file
    if os.path.isdir(pred_path):
        continue
    if pred_file[-4:] == ".png":
        continue
    print(pred_path)
    pred_3 = pd.read_csv(pred_path, header=None)
    pred_3 = pred_3.values
    plt.figure(figsize=(4, 4), dpi=300)
    class_num = ["Noninfected", "Bacterial", "Viral"]
    for i in [1, 2, 0]:
        label_one = (label == i) * 1
        # pred_one = (pred == i) * 1
        pred_one = pred_3[:, i]
        # pred_one = (pred_3.argmax(axis=1) == i)

        # print(label_one.shape)
        # print(pred_one.shape)
        pred_one = pred_one.flatten()

        fpr, tpr, _ = roc_curve(label_one, pred_one)
        roc_auc = auc(fpr, tpr)
        lw = 1
        plt.plot(fpr, tpr, lw=lw, label=f'{class_num[i]} (AUC=%0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        # plt.title('ROC Curve')
        plt.grid()
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{path_png}{pred_file[:-4]}.png')
    plt.show()

    plt.close()
    # input()
