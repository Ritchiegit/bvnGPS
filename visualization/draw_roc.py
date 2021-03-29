import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
import matplotlib
import glob
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
os.chdir("..")

path_model = "results/final_model_results/20210326_external2_1_model_selected_0328/pred_result/2/only_21802_57065None/"
dataset = "only_21802_57065"
type_part_dataset = None
# path_model = "results/final_model_results/20210326_external2_1_model_selected_0328/pred_result/2/all_exclude_21802_570650.3/"
# dataset = "all_exclude_21802_57065"
# type_part_dataset = "0.3"
# path_model = "results/final_model_results/20210326_external2_1_model_selected_0328/pred_result/2/all_exclude_21802_570650.7/"
# dataset = "all_exclude_21802_57065"
# type_part_dataset = "0.7"



path_png = f'{path_model}/png/'
if not os.path.exists(path_png):
    os.makedirs(path_png)

dataset_random_state = 1

# pred = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/oneCART.model_pickle.csv", header=None)
# pred_3 = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/threeCART.model_pickle.csv", header=None)
# pred = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/onelda.model_pickle.csv", header=None)
# pred_3 = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/threelda.model_pickle.csv", header=None)
# pred = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/oneSVM_klinear_dovr_C0.1.model_pickle.csv", header=None)
# pred_3 = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/threeSVM_klinear_dovr_C0.1.model_pickle.csv", header=None)
# pred = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/oneNeuralNetworkMSE_optAdagrad_h768_lr0.0005_val0.054926375896610015_epoch262.pth.csv", header=None)
# pred_3 = pd.read_csv("results/final_model_results/20210326_external2_1_model_selected/pred_result/2/threeNeuralNetworkMSE_optAdagrad_h768_lr0.0005_val0.054926375896610015_epoch262.pth.csv", header=None)
# pred = pred.values
# pred_3 = pred_3.values
# print(pred.shape)
# print(pred_3.shape)
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
pred_file_list = os.listdir(path_model)
for pred_file in pred_file_list:
    pred_path = path_model + "/" + pred_file
    if os.path.isdir(pred_path):
        continue
    print(pred_path)
    pred_3 = pd.read_csv(pred_path, header=None)
    pred_3 = pred_3.values
    plt.figure(figsize=(4, 4), dpi=300)

    for i in range(3):
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
        plt.plot(fpr, tpr, lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC Curve')
        plt.grid()
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{path_png}{pred_file[:-4]}.png')
    plt.show()

    plt.close()
    # input()
