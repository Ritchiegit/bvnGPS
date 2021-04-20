import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib
import glob
from scipy import stats
import numpy as np
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
os.chdir("..")
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw

front_str = "20210416_2"

dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.7"
# dataset = "only_21802_57065"
# type_part_dataset = None

threshold = 0.3
file_to_save_name = f"{dataset}{type_part_dataset}_{threshold}.csv"

# pre_result_total_path = "results/final_model_results/20210326_for_all_sensitivity/pred_result/"
# pre_result_total_path = "results/final_model_results/20210408_1_e2_raw63990_iPAGE/pred_result/"
pre_result_total_path = "results/final_model_results/20210416_2/pred_result/"
random_seed_filename_list = os.listdir(pre_result_total_path)
print(random_seed_filename_list)
path_acc_sensitivity_specificity = f'{pre_result_total_path}/acc_s_s/'
dataset_random_state = 1
if not os.path.exists(path_acc_sensitivity_specificity):
    os.makedirs(path_acc_sensitivity_specificity)
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

df_list = []

for random_seed_filename in random_seed_filename_list:
    if random_seed_filename[:len(front_str)] != front_str:
    # if random_seed_filename[:8] != "20210408":
        continue
    path_model = f"{pre_result_total_path}/{random_seed_filename}/{dataset}{type_part_dataset}/"
    print(path_model)
    # pred_path_list = glob.glob(path_model + "*")
    pred_file_list = os.listdir(path_model)
    final_print = []
    pred_file_list.sort()
    for pred_file in pred_file_list:
        pred_path = path_model + "/" + pred_file
        if os.path.isdir(pred_path):
            continue
        print(pred_path)
        pred_3 = pd.read_csv(pred_path, header=None)
        pred_3 = pred_3.values
        # plt.figure(figsize=(4, 4), dpi=300)
        class_num = ["Noninfected", "Bacterial", "Viral"]
        # if check_front_of_name(pred_file, "threeCART"):
        #     pred_file = "CART"
        # elif check_front_of_name(pred_file, "threelda"):
        #     pred_file = "RandomForest"
        # elif check_front_of_name(pred_file, "threeNeural"):
        #     pred_file = "NeuralNetwork"
        # elif check_front_of_name(pred_file, "threeSVM"):
        #     pred_file = "SVM"
        # else:
        #     print("检查所存模型输出文件夹名称")
        #     input()
        for i in [1, 2, 0]:
            label_one = (label == i) * 1
            # pred_one = (pred == i) * 1
            pred_one = pred_3[:, i]
            # pred_one = (pred_3.argmax(axis=1) == i)

            # print(label_one.shape)
            # print(pred_one.shape)
            pred_one = pred_one.flatten()
            pred_one = (pred_one > threshold) * 1
            from sklearn.metrics import confusion_matrix

            tn, fp, fn, tp = confusion_matrix(label_one, pred_one).ravel()
            acc = (tp + tn) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            save_one = (f"{pred_file}_{class_num[i]}", acc, sensitivity, specificity, tn, fp, fn, tp)
            # print(save_one)
            final_print.append(save_one)
        #     fpr, tpr, _ = roc_curve(label_one, pred_one)
        #     roc_auc = auc(fpr, tpr)
        #     lw = 1
        #     plt.plot(fpr, tpr, lw=lw, label=f'{class_num[i]} (AUC=%0.2f)' % roc_auc)
        #     plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        #     # plt.title('ROC Curve')
        #     plt.grid()
        #     plt.legend(loc="lower right")
        #     plt.tight_layout()
        #     # plt.savefig(f'{path_png}{pred_file[:-4]}.png')
        # plt.show()

        # plt.close()
        # input()
    df = pd.DataFrame(final_print)
    df.set_index([0], inplace=True)
    df_list.append(df)
print(len(df_list))
final_result_nptensor = np.dstack(df_list)
print("final_result_nptensor.shape", final_result_nptensor.shape)
# df.to_csv(path_acc_sensitivity_specificity+file_to_save_name)
mean = final_result_nptensor.mean(2)
# print(mean)
print("mean.shape", mean.shape)
std = final_result_nptensor.std(axis=2, ddof=1)
# print(std)
print("std.shape", std.shape)
conf_interval = stats.norm.interval(0.95, loc=mean, scale=std)
# print(conf_interval)

for_selecting = np.isnan(conf_interval[0])
# print(for_selecting)
mean_selected = for_selecting * mean
print("mean_selected.shape", mean_selected.shape)
new_conf_interval = [conf_interval[0], conf_interval[1]]
new_conf_interval[0][np.isnan(new_conf_interval[0])] = 0
new_conf_interval[1][np.isnan(new_conf_interval[1])] = 0
new_conf_interval[0] += mean_selected
new_conf_interval[1] += mean_selected

new_conf_interval_concated = np.hstack((mean, new_conf_interval[0], new_conf_interval[1]))
print("new_conf_interval_concated.shape", new_conf_interval_concated.shape)
conf_interval_concated_pd = pd.DataFrame(new_conf_interval_concated)
conf_interval_concated_pd.index = df.index
conf_interval_concated_pd.columns = ["acc", "sen", "spec", "tn", "fp", "fn", "tp",
                                     "acc1", "sen1", "spec1", "tn1", "fp1", "fn1", "tp1",
                                     "acc2", "sen2", "spec2", "tn2", "fp2", "fn2", "tp2",
                                     ]
conf_interval_concated_pd.to_csv(path_acc_sensitivity_specificity+file_to_save_name)