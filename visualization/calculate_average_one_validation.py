import pandas as pd
import glob

source_folder = "../results/final_model_results/20210408_1_e2_raw63990_iPAGE/predict_onall_exclude_21802_570650.7_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210408_1_e2_raw63990_iPAGE/predict_onall_exclude_21802_570650.7_validation/0.7average.csv"
#
# source_folder = "../results/final_model_results/20210408_1_e2_raw63990_iPAGE/predict_ononly_21802_57065None_validation/2*.csv"
# final_average_csv_path = "../results/final_model_results/20210408_1_e2_raw63990_iPAGE/predict_ononly_21802_57065None_validation/only_21802_57065.csv"
source_folder = "../results/final_model_results/20210416_1/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_1/test_average.csv"
# source_folder = "../results/final_model_results/20210416_2/2*.csv"
# final_average_csv_path = "../results/final_model_results/20210416_2/test_average.csv"
source_folder = "../results/final_model_results/20210416_1/predict_ononly_21802_57065None_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_1/val_0416_1_average.csv"
source_folder = "../results/final_model_results/20210416_2/predict_ononly_21802_57065None_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_2/val_0416_2_average.csv"

source_folder = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.7_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.7_validation/train_0416_2_average.csv"

source_folder = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.3_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_2/predict_onall_exclude2_6269_1_raw639900.3_validation/test_0416_2_average.csv"


source_folder = "../results/final_model_results/20210416_2/predict_ononly_21802_57065None_validation/2*.csv"
final_average_csv_path = "../results/final_model_results/20210416_2/predict_ononly_21802_57065None_validation/val_0416_2_new_average.csv"


final_result_list = glob.glob(source_folder)
print(len(final_result_list))
print(final_result_list)
final_result_acc = 0
acc_num = 0
for final_result_list_each in final_result_list:
    # print(final_result_list_each)
    final_each_pd = pd.read_csv(final_result_list_each, header=None, index_col=0)
    # if 'lightgbm' in final_each_pd.index:
    #     final_each_pd.drop(['CART', 'RandomForest', 'lightgbm'], axis=0, inplace=True)
    # else:
    #     final_each_pd.drop(['CART', 'RandomForest'], axis=0, inplace=True)
    # print(final_each_pd)
    # print(final_each_pd[1])
    # input()
    # final_each_pd = final_each_pd.values[:330]  # TODO 1
    final_each_pd = final_each_pd.values  # TODO 1

    if final_each_pd.shape[0] == 120:
        print(final_result_list_each)
        print(final_each_pd.shape)
        final_result_acc += final_each_pd
        acc_num += 1

# input()
final_result_aver = final_result_acc / acc_num
print(final_result_aver)
final_result_aver = pd.DataFrame(final_result_aver)  # TODO 2
# final_result_aver.to_csv(final_average_csv_path)

final_each_pd = pd.read_csv(final_result_list[11], header=None, index_col=0)
final_result_aver.index = final_each_pd.index
final_result_aver.to_csv(final_average_csv_path)
