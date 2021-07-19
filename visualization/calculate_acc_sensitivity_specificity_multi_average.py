import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
os.chdir("..")
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw

front_str = "20210512_1"
pre_result_total_path = "results/final_model_results/20210512_1/pred_result/"

dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"
# dataset = "all_exclude2_6269_1_raw63990"
# type_part_dataset = "0.7"
# dataset = "only_21802_57065"
# type_part_dataset = None

threshold = 0.3
file_to_save_name = f"{dataset}{type_part_dataset}_{threshold}.csv"

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
        continue
    if random_seed_filename[-5:] != "model":  # 需要将存acc的文件夹筛选掉
        continue
    path_model = f"{pre_result_total_path}/{random_seed_filename}/{dataset}{type_part_dataset}/"
    pred_file_list = os.listdir(path_model)
    final_print = []
    pred_file_list.sort()
    for pred_file in pred_file_list:
        pred_path = path_model + "/" + pred_file
        if os.path.isdir(pred_path):
            continue
        if pred_path[-4:] != ".csv":
            continue
        print(pred_path)
        pred_3 = pd.read_csv(pred_path, header=None)
        pred_3 = pred_3.values
        class_num = ["Noninfected", "Bacterial", "Viral"]
        for i in [1, 2, 0]:
            label_one = (label == i) * 1
            pred_one = pred_3[:, i]
            # pred_one = (pred_3.argmax(axis=1) == i)
            pred_one = pred_one.flatten()
            pred_one = (pred_one > threshold) * 1

            tn, fp, fn, tp = confusion_matrix(label_one, pred_one).ravel()
            acc = (tp + tn) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            save_one = (f"{pred_file}_{class_num[i]}", acc, sensitivity, specificity, tn, fp, fn, tp)
            # print(save_one)
            final_print.append(save_one)
    df = pd.DataFrame(final_print)
    df.set_index([0], inplace=True)
    df_list.append(df)
print(len(df_list))
final_result_nptensor = np.dstack(df_list)
print("final_result_nptensor.shape", final_result_nptensor.shape)
mean = final_result_nptensor.mean(2)
print("mean.shape", mean.shape)
std = final_result_nptensor.std(axis=2, ddof=1)
print("std.shape", std.shape)
conf_interval = stats.norm.interval(0.95, loc=mean, scale=std)
for_selecting = np.isnan(conf_interval[0])
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