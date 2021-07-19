import numpy as np
from scipy import stats

import pandas as pd
import glob

source_folder = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.7_validation/2*.csv"
final_ci_csv_path = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.7_validation/train_0416_3_CI.csv"

# source_folder = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.3_validation/2*.csv"
# final_ci_csv_path = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.3_validation/test_0416_3_CI.csv"

# source_folder = "../results/final_model_results/20210416_3_smx/predict_ononly_21802_57065None_validation/2*.csv"
# final_ci_csv_path = "../results/final_model_results/20210416_3_smx/predict_ononly_21802_57065None_validation/val_0416_3_CI.csv"


final_result_list = glob.glob(source_folder)
print(len(final_result_list))
print(final_result_list)
final_result_acc_list = []
for final_result_list_each in final_result_list:
    print(final_result_list_each)
    final_each_pd = pd.read_csv(final_result_list_each, header=None, index_col=0)
    final_each_pd = final_each_pd.values[:120]  # TODO 1
    final_result_acc_list.append(final_each_pd)
final_result_nptensor = np.dstack(final_result_acc_list)
mean = final_result_nptensor.mean(2)
std = final_result_nptensor.std(axis=2, ddof=1)
conf_interval = stats.norm.interval(0.95, loc=mean, scale=std)

print("*"*50)
for_selecting = np.isnan(conf_interval[0])
mean_selected = for_selecting * mean
new_conf_interval = [conf_interval[0], conf_interval[1]]
new_conf_interval[0][np.isnan(new_conf_interval[0])] = 0
new_conf_interval[1][np.isnan(new_conf_interval[1])] = 0
new_conf_interval[0] += mean_selected
new_conf_interval[1] += mean_selected
final_each_pd = pd.read_csv(final_result_list[0], header=None, index_col=0)
new_conf_interval_concated = np.hstack(new_conf_interval)
print(new_conf_interval_concated.shape)
print(new_conf_interval_concated)
conf_interval_concated_pd = pd.DataFrame(new_conf_interval_concated)
conf_interval_concated_pd.index = final_each_pd.index
conf_interval_concated_pd.to_csv(final_ci_csv_path)