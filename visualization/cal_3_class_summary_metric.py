from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
os.chdir("..")
from data_processing.process_data_label import get_label_multilabel
from load_data.load_data_raw import load_data_raw

front_str = "20210416_3_smx"

dataset = "all_exclude2_6269_1_raw63990"
type_part_dataset = "0.3"
# dataset = "all_exclude2_6269_1_raw63990"
# type_part_dataset = "0.7"
# dataset = "only_21802_57065"
# type_part_dataset = None

threshold = 0.3
file_to_save_name = f"3_class_acc_p_{dataset}{type_part_dataset}_{threshold}.csv"

pre_result_total_path = "results/final_model_results/20210416_3_smx/pred_result/"
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
        continue
    if random_seed_filename[-5:] != "model":
        continue
    path_model = f"{pre_result_total_path}/{random_seed_filename}/{dataset}{type_part_dataset}/"
    print("path_model", path_model)
    pred_file_list = os.listdir(path_model)
    final_print = []
    pred_file_list.sort()  # important
    for pred_file in pred_file_list:  # The output of a single model csv A single csv calculates an acc
        pred_path = path_model + "/" + pred_file
        if os.path.isdir(pred_path):
            continue
        if pred_path[-4:] != ".csv":
            continue
        pred_3 = pd.read_csv(pred_path, header=None)
        pred_3 = pred_3.values
        pred_3_argmax = np.argmax(pred_3, axis=1)
        acc_1 = sklearn.metrics.accuracy_score(label, pred_3_argmax)
        p_macro = sklearn.metrics.precision_score(label, pred_3_argmax, average="macro")
        p_micro = sklearn.metrics.precision_score(label, pred_3_argmax, average="micro")
        p_weighted = sklearn.metrics.precision_score(label, pred_3_argmax, average="weighted")
        r_macro = sklearn.metrics.recall_score(label, pred_3_argmax, average="macro")
        r_micro = sklearn.metrics.recall_score(label, pred_3_argmax, average="micro")
        r_weighted = sklearn.metrics.recall_score(label, pred_3_argmax, average="weighted")
        f1_macro = sklearn.metrics.f1_score(label, pred_3_argmax, average="macro")
        f1_micro = sklearn.metrics.f1_score(label, pred_3_argmax, average="micro")
        f1_weighted = sklearn.metrics.f1_score(label, pred_3_argmax, average="weighted")
        kappa_score = sklearn.metrics.cohen_kappa_score(label, pred_3_argmax)
        hamming_loss = sklearn.metrics.hamming_loss(label, pred_3_argmax)
        confusion_matrix = sklearn.metrics.confusion_matrix(label, pred_3_argmax)
        print(confusion_matrix)


        save_one = (f"{pred_file}", acc_1, p_macro, p_micro, p_weighted, r_macro, r_micro, r_weighted, f1_micro, f1_micro, f1_weighted, kappa_score, hamming_loss)
        final_print.append(save_one)
    # Multiple csv (multiple models) results are arranged to get a summary of all models final_print
    df = pd.DataFrame(final_print)
    df.set_index([0], inplace=True)
    df_list.append(df)  # Summary of all random numbers
print(len(df_list))
final_result_nptensor = np.dstack(df_list)
# print("final_result_nptensor.shape", final_result_nptensor.shape)
final_result_mean = final_result_nptensor.mean(2)
print("final_result_mean.shape", final_result_mean.shape)
final_result_mean_pd = pd.DataFrame(final_result_mean)
final_result_mean_pd.columns = ["acc_1", "p_macro", "p_micro", "p_weighted", "r_macro", "r_micro", "r_weighted", "f1_micro", "f1_micro", "f1_weighted", "kappa_score", "hamming_loss"]
final_result_mean_pd.index = df.index
print(final_result_mean_pd)
# final_result_mean_pd.to_csv(path_acc_sensitivity_specificity+file_to_save_name)
