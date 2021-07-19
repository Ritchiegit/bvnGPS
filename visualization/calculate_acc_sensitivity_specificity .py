import matplotlib.pyplot as plt
import pandas as pd
import os
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
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


file_to_save_name = f"{dataset}{type_part_dataset}.csv"
path_acc_sensitivity_specificity = f'{path_model}/acc_s_s/'
if not os.path.exists(path_acc_sensitivity_specificity):
    os.makedirs(path_acc_sensitivity_specificity)

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
pred_file_list = os.listdir(path_model)
final_print = []

for pred_file in pred_file_list:
    pred_path = path_model + "/" + pred_file
    if os.path.isdir(pred_path):
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
        threshold = 0.5
        pred_one = (pred_one > threshold) * 1
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(label_one, pred_one).ravel()
        acc = (tp + tn) / (tn + fp + fn + tp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        save_one = (pred_file[:-4], class_num[i], acc, sensitivity, specificity, tn, fp, fn, tp)
        print(save_one)
        final_print.append(save_one)

df = pd.DataFrame(final_print)
df.to_csv(path_acc_sensitivity_specificity+file_to_save_name)