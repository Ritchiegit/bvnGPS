import pandas as pd
import glob
source_folder = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.7_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.7_validation/train_0416_3_average.csv"

# source_folder = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.3_validation/2*.csv"
# final_average_csv_path = "../results/final_model_results/20210416_3_smx/predict_onall_exclude2_6269_1_raw639900.3_validation/test_0416_3_average.csv"

# source_folder = "../results/final_model_results/20210416_3_smx/predict_ononly_21802_57065None_validation/2*.csv"
# final_average_csv_path = "../results/final_model_results/20210416_3_smx/predict_ononly_21802_57065None_validation/val_0416_3_average.csv"


final_result_list = glob.glob(source_folder)
print(len(final_result_list))
print(final_result_list)
final_result_acc = 0
acc_num = 0
for final_result_list_each in final_result_list:
    # print(final_result_list_each)
    final_each_pd = pd.read_csv(final_result_list_each, header=None, index_col=0)
    final_each_pd = final_each_pd.values

    if final_each_pd.shape[0] == 120:
        print(final_result_list_each)
        print(final_each_pd.shape)
        final_result_acc += final_each_pd
        acc_num += 1

# input()
final_result_aver = final_result_acc / acc_num
print(final_result_aver)
final_result_aver = pd.DataFrame(final_result_aver)

final_each_pd = pd.read_csv(final_result_list[11], header=None, index_col=0)
final_result_aver.index = final_each_pd.index
final_result_aver.to_csv(final_average_csv_path)
