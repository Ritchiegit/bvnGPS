import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os
# source_folder_train = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.7_validation/"
# source_folder_test = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.3_validation/"
# source_folder_val = "../results/final_model_results/20210416_2/predict_ononly_21802_57065None_validation/"
dataset_sig = "20210416_3_smx"
source_folder_train = f"../results/final_model_results/{dataset_sig}/predict_onall_exclude2_6269_1_raw639900.7_validation/"
source_folder_test = f"../results/final_model_results/{dataset_sig}/predict_onall_exclude2_6269_1_raw639900.3_validation/"
source_folder_val = f"../results/final_model_results/{dataset_sig}/predict_ononly_21802_57065None_validation/"
# final_average_csv_path = "../results/final_model_results/20210326_external2_1_iPAGE_all_exclude_21802_57065/predict_ononly_21802_57065None_validation/10.csv"

# 需要读入的数据

# fig = plt.figure(figsize=(8, 8))
fig, all_ax = plt.subplots(1, 4, figsize=(8, 8))
cbar_ax = fig.add_axes([.91, .3, .03, .4])
source_folder_all = [source_folder_train, source_folder_test, source_folder_val]


index_list = [0, 119, 80, 36]
model_name = ["Decision Tree", "Random Forest", "SVM", "Neural Network"]
for i in range(4):
    csv_save_path = f"{dataset_sig}auc_summary_{model_name[i]}.csv"
    if not os.path.exists(csv_save_path):
        index = index_list[i]
        source_output_50_list = []
        for source_folder_one in source_folder_all:
            csv_path = glob.glob(source_folder_one+"2*.csv")
            # print(csv_path)
            one_model_result_50_list = []
            for csv_path_one in csv_path:
                csv_one = pd.read_csv(csv_path_one, header=None, index_col=0)
                csv_one.columns = ["Non-infected", "Bacterial", "Viral", "epoches"]
                # print(csv_one)
                # print(csv_one.columns)

                model_csv_selected = csv_one[["Bacterial", "Viral", "Non-infected"]].iloc[[index]]  # .iloc[[0, 119, 80, 36]]
                # print(model_csv_selected)
                one_model_result_50_list.append(model_csv_selected)
            one_model_result_50 = pd.concat(one_model_result_50_list, axis=0)
            # print(one_model_result_50)
            #a = one_model_result_50.loc["CART.model_pickle"]
            # print(one_model_result_50.head(3))
            # print(np.average(one_model_result_50, axis=0))
            source_output_50_list.append(one_model_result_50)
        # print(source_output_50_list)
        source_output_50_train_test_val = pd.concat(source_output_50_list, axis=1)
        print(source_output_50_train_test_val)
        source_output_50_train_test_val.columns = ["train_b", "train_v", "train_n",
                                                   "test_b", "test_v", "test_n",
                                                   "val_b", "val_v", "val_n", ]
        print(source_output_50_train_test_val)# yticklabels=source_output_50_train_test_val.columns
        source_output_50_train_test_val.to_csv(csv_save_path)
    else:
        source_output_50_train_test_val = pd.read_csv(csv_save_path, header=0, index_col=0)

    plt.subplot(1, 4, i + 1)
    ax = sns.heatmap(source_output_50_train_test_val, yticklabels=False, xticklabels=source_output_50_train_test_val.columns,
                     annot=True, fmt='.2f', annot_kws={"size": 4},
                     vmin=0.5, vmax=1, cmap="Reds", cbar=True, cbar_ax=cbar_ax)  # Greys	PuRd	RdPu	OrRd	Reds	YlOrRd
    plt.xticks(fontsize=7)
    ax.tick_params(left=False)
    ax.set_ylabel('')
    # plt.suptitle(f"{model_name[i]}")
    all_ax[i].set_title(f"{model_name[i]}")

fig.tight_layout(rect=[0, 0, .9, 1])

plt.savefig(f'{dataset_sig}_heatmap_1.png', transparent=True, dpi=800)
plt.show()