import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw
from sklearn.metrics import confusion_matrix
os.chdir("..")
average = "micro"
# average = "weighted"

front_str = "20210512_1"

pre_result_total_path = "results/final_model_results/20210512_1/pred_result/"

dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"
dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.7"
dataset = "only_21802_57065"
type_part_dataset = None
threshold = 0.3
file_to_save_name = f"{front_str}_final_simple_summary_acc_sens_spec_{average}_{dataset}{type_part_dataset}_{threshold}.csv"

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
else:
    assert type_part_dataset is None
label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
label_0 = (label == 0) * 1
label_1 = (label == 1) * 1
label_2 = (label == 2) * 1
num_label_0 = sum(label_0)
num_label_1 = sum(label_1)
num_label_2 = sum(label_2)
ratio_label_0 = num_label_0/(num_label_0+num_label_1+num_label_2)
ratio_label_1 = num_label_1/(num_label_0+num_label_1+num_label_2)
ratio_label_2 = num_label_2/(num_label_0+num_label_1+num_label_2)
print(num_label_0)
print(num_label_1)
print(num_label_2)
print(ratio_label_0)
print(ratio_label_1)
print(ratio_label_2)
print(average)
df_list = []
for random_seed_filename in random_seed_filename_list:
    if random_seed_filename[:len(front_str)] != front_str:
        continue
    if random_seed_filename[-5:] != "model":
        continue
    path_model = f"{pre_result_total_path}/{random_seed_filename}/{dataset}{type_part_dataset}/"
    print("line 64", path_model)
    # pred_path_list = glob.glob(path_model + "*")
    pred_file_list = os.listdir(path_model)
    final_print = []
    pred_file_list.sort()
    for pred_file in pred_file_list:
        pred_path = path_model + "/" + pred_file
        if os.path.isdir(pred_path):
            continue
        if pred_path[-4:] != ".csv":
            continue
        pred_3 = pd.read_csv(pred_path, header=None)
        pred_3 = pred_3.values
        tn_fp_fn_tp_list_3 = []
        for i in [1, 2, 0]:
            label_one = (label == i) * 1
            pred_one = pred_3[:, i]
            pred_one = pred_one.flatten()
            pred_one = (pred_one > threshold) * 1
            tn, fp, fn, tp = confusion_matrix(label_one, pred_one).ravel()
            tn_fp_fn_tp_list_3.append((tn, fp, fn, tp))
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        tp_sum = 0
        if average == "micro":
            for i in [0, 1, 2]:
                tn_sum += tn_fp_fn_tp_list_3[i][0]
                fp_sum += tn_fp_fn_tp_list_3[i][1]
                fn_sum += tn_fp_fn_tp_list_3[i][2]
                tp_sum += tn_fp_fn_tp_list_3[i][3]
            acc_average = (tp_sum + tn_sum) / (tn_sum + fp_sum + fn_sum + tp_sum)
            sens_average = tp_sum / (tp_sum + fn_sum)
            spec_average = tn_sum / (tn_sum + fp_sum)
        elif average == "weighted":
            acc_list_3 = []
            sens_list_3 = []
            spec_list_3 = []
            for tn, fp, fn, tp in tn_fp_fn_tp_list_3:
                weight = (tp + fn) / (tn + fp + fn + tp)
                acc = (tp + tn) / (tn + fp + fn + tp) * weight
                sensitivity = tp / (tp + fn) * weight
                specificity = tn / (tn + fp) * weight
                acc_list_3.append(acc)
                sens_list_3.append(sensitivity)
                spec_list_3.append(specificity)
                # acc = (tp + tn) / (tn + fp + fn + tp)
                # sensitivity = tp / (tp + fn)
                # specificity = tn / (tn + fp)
                # acc_list_3.append(acc)
                # sens_list_3.append(sensitivity)
                # spec_list_3.append(specificity)
            # acc_average = acc_list_3[0]*ratio_label_1 + acc_list_3[1]*ratio_label_2 + acc_list_3[2]*ratio_label_0
            # sens_average = sens_list_3[0]*ratio_label_1 + sens_list_3[1]*ratio_label_2 + sens_list_3[2]*ratio_label_0
            # spec_average = spec_list_3[0]*ratio_label_1 + spec_list_3[1]*ratio_label_2 + spec_list_3[2]*ratio_label_0
            acc_average = acc_list_3[0] + acc_list_3[1] + acc_list_3[2]
            sens_average = sens_list_3[0] + sens_list_3[1] + sens_list_3[2]
            spec_average = spec_list_3[0] + spec_list_3[1] + spec_list_3[2]

        else:
            print("average type is wrong")
            exit(1)
        save_one = (f"{pred_file}", acc_average, sens_average, spec_average,
                    tn_fp_fn_tp_list_3[0][0], tn_fp_fn_tp_list_3[0][1], tn_fp_fn_tp_list_3[0][2], tn_fp_fn_tp_list_3[0][3],
                    tn_fp_fn_tp_list_3[1][0], tn_fp_fn_tp_list_3[1][1], tn_fp_fn_tp_list_3[1][2], tn_fp_fn_tp_list_3[1][3],
                    tn_fp_fn_tp_list_3[2][0], tn_fp_fn_tp_list_3[2][1], tn_fp_fn_tp_list_3[2][2], tn_fp_fn_tp_list_3[2][3])
        final_print.append(save_one)
    df = pd.DataFrame(final_print)
    df.set_index([0], inplace=True)
    df_list.append(df)
print(len(df_list))
final_result_nptensor = np.dstack(df_list)
print("final_result_nptensor.shape", final_result_nptensor.shape)
final_result_mean = final_result_nptensor.mean(2)
print("final_result_mean.shape", final_result_mean.shape)
final_result_mean_pd = pd.DataFrame(final_result_mean)
final_result_mean_pd.columns = ["acc_average", "sens_average", "spec_average", "tn0", "fp0", "fn0", "tp0", "tn1", "fp1", "fn1", "tp1", "tn2", "fp2", "fn2", "tp2"]
final_result_mean_pd.index = df.index
print(final_result_mean_pd)
final_result_mean_pd.to_csv(path_acc_sensitivity_specificity+file_to_save_name)