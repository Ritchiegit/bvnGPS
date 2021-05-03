import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


"""
# https://blog.csdn.net/jiajiren11/article/details/90400595
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

N = 10000
x = np.random.normal(0, 1, N)
# ddof取值为1是因为在统计学中样本的标准偏差除的是(N-1)而不是N，统计学中的标准偏差除的是N
# SciPy中的std计算默认是采用统计学中标准差的计算方式
# mean, std = x.mean(), x.std(ddof=1)
# mean, std = 0, 1
a1 = np.ones((4, 3, 1))
a2 = 2 * np.ones((4, 3, 1))
a3 = 3 * np.ones((4, 3, 1))
a = np.dstack((a1, a2, a3))
print(a.shape)

mean = a.mean(2)
print(mean)
std = a.std(axis=2, ddof=1)
print(std)
print(mean, std)
# 计算置信区间
# 这里的0.9是置信水平
conf_intveral = stats.norm.interval(0.6827, loc=mean, scale=std)
print(conf_intveral)
conf_intveral = stats.norm.interval(0.9545, loc=mean, scale=std)
print(conf_intveral)
conf_intveral = stats.norm.interval(0.9973, loc=mean, scale=std)
print(conf_intveral)
"""
# input mean, std

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