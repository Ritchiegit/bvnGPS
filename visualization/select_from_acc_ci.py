import pandas as pd
import glob
acc_path = "/mnt/qzli_hdd/Bio/ovr_ipage2biomarker_summaryandpredict/results/final_model_results/20210416_2/pred_result/acc_s_s/"

csv_list = glob.glob(acc_path+"all_exclude2_6269_1_raw639900.3_*.csv")
csv_list.sort()
nn_selected_list = []
for csv_one_path in csv_list:
    csv_one = pd.read_csv(csv_one_path)
    nn_selected = csv_one.iloc[108:111][["0", "acc", "sen", "spec", "acc1", "sen1", "spec1", "acc2", "sen2", "spec2"]]
    # print(nn_selected)
    nn_selected_list.append(nn_selected)
    print(csv_one_path)
nn_selected_pd = pd.concat(nn_selected_list, axis=0)
print(nn_selected_pd)
nn_selected_pd.to_csv(f"{acc_path}acc_ci_summary.csv")