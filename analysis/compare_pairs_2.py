# D:\Bio\workspace\ovr_ipage2biomarker_summaryandpredict\results\20210228_pair_select
import glob
import pandas as pd
from utils import load_list_of_tuple

# pair_list_glob = glob.glob("../results/20210228_pair_select/*/*/pair_after_lasso.csv")
# pair_list_glob_0302 = glob.glob("../results/20210302_pair_select/*/*/pair_after_lasso.csv")
#
# print(pair_list_glob)
# acc_tuple = 0
# for i, (pair_glob_0228, pair_glob_0302) in enumerate(zip(pair_list_glob, pair_list_glob_0302)):
#     print(pair_glob_0228)
#     print(pair_glob_0302)
#     # if i == 0:
#     #     acc_tuple = set(load_list_of_tuple(pair_glob))
#     # print(pair_glob)
#     # pair_tuple = set(load_list_of_tuple(pair_glob))
#     # acc_tuple = acc_tuple&pair_tuple
#     # # print(acc_tuple)
#     # # input()


# # a_pd = pd.read_csv(a_path, index_col=None, header=None)
# # b_pd = pd.read_csv(a_path, index_col=None, header=None)
# # print(a_pd)
# # print(b_pd)
# pair_a = load_list_of_tuple(a_path)
# pair_b = load_list_of_tuple(b_path)
# print(pair_a)
# # print(pair_b)
def two_2_str(a12):
    (a1, a2) = a12
    return str(a1), str(a2)
#
# # c = two_2_str(pair_a[0])
# a_str = list(map(two_2_str, pair_a))
# print(a_str)
# b_str = list(map(two_2_str, pair_b))
# print(b_str)
# count = 0
# for rna_1, rna_2 in b_str:
#     if (rna_1, rna_2) in a_str or (rna_2, rna_1) in a_str:
#         count += 1
#
# print(count)
# count = 0
# for rna_1, rna_2 in b_str:
#     if (rna_1, rna_2) in a_str:
#         count += 1
#
# print(count)
# count = 0
# for rna_1, rna_2 in b_str:
#     if (rna_2, rna_1) in a_str:
#         count += 1
#
# print(count)

def compare_gene_pair(path_a, path_b):
    pair_a = load_list_of_tuple(path_a)
    pair_b = load_list_of_tuple(path_b)
    pair_a_len = len(pair_a)
    pair_b_len = len(pair_b)
    print(pair_a_len, pair_b_len)
    if pair_a_len != pair_b_len:
        print("len is not equal")
        # return
    a_str = list(map(two_2_str, pair_a))
    b_str = list(map(two_2_str, pair_b))
    count = 0
    for rna_1, rna_2 in b_str:
        if (rna_1, rna_2) in a_str or (rna_2, rna_1) in a_str:
            count += 1
    print(count)
    if count == pair_a_len:
        print("Equal")
        return True
    else:
        print("no Equal")
        return False


# a_path = "../results/20210303_t_13_nosortset_pair_iPAGE_coco_nc2020_seed53_loc20210304_105150/biomarker/pair_after_lasso_2.csv"
# b_path = "../results/20210303_t_13pair_iPAGE_coco_nc2020_seed53_loc20210304_103612/biomarker/pair_after_lasso_2.csv"
# a_path = "../results/20210303_t_14_nosortset_pair_iPAGE_coco_nc2020_seed53_loc20210304_141741/biomarker/pair_after_lasso.csv"
# b_path = "../results/20210303_t_15_nosort_pathway_set_pair_iPAGE_coco_nc2020_seed53_loc20210304_141951/biomarker/pair_after_lasso.csv"
# a_path = "../results/20210303_t_13_nosortset_pair_iPAGE_coco_nc2020_seed53_loc20210304_105150/biomarker/pair_after_lasso_2.csv"
# b_path = "../results/20210303_t_16_nosortset_63_pair_iPAGE_coco_nc2020_seed53_loc20210304_144113/biomarker/pair_after_lasso_2.csv"
# a_path = "../results/20210303_t_21_63sorted_pair_iPAGE_coco_nc2020_seed53_loc20210304_150838/biomarker/pair_after_lasso_0.csv"
# b_path = "../results/20210303_t_22_63sorted_pair_iPAGE_coco_nc2020_seed58_loc20210304_150951/biomarker/pair_after_lasso_0.csv"
# a_path = f"../results/20210304_1_pair_69_unlock_first_set_iPAGE_coco_nc2020_seed69_loc20210304_213515/biomarker/pair_after_lasso_0.csv"
# b_path = f"../results/20210304_1_pair_iPAGE_coco_nc2020_seed55_loc20210304_153737/biomarker/pair_after_lasso_0.csv"
# a_path = f"../results/20210304_1_pair_69_unlock_lasso_iPAGE_coco_nc2020_seed69_loc20210304_213334/biomarker/pair_after_lasso_{i}.csv"
# b_path = f"../results/20210304_1_pair_iPAGE_coco_nc2020_seed55_loc20210304_153737/biomarker/pair_after_lasso_{i}.csv"

# file_a = ""
# file_b = ""
#
# for i in range(3):
#     a_path = f"../results/20210304_1_pair_69_unlock_lasso_iPAGE_coco_nc2020_seed69_loc20210304_213334/biomarker/pair_after_lasso_{i}.csv"
#     b_path = f"../results/20210304_1_pair_iPAGE_coco_nc2020_seed55_loc20210304_153737/biomarker/pair_after_lasso_{i}.csv"
#     compare_gene_pair(a_path, b_path)

def folder_gene_compare(folder_a, folder_b):
    print("#"*50)

    print(folder_a)
    print(folder_b)
    for i in range(3):
        print("**************", i, "****************")
        a_path = f"../results/{folder_a}/biomarker/pair_after_lasso_{i}.csv"
        b_path = f"../results/{folder_b}/biomarker/pair_after_lasso_{i}.csv"
        compare_gene_pair(a_path, b_path)

folder_a = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS1_loc20210304_224937"
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS3_loc20210304_222432"  # 13 13 19
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS5_loc20210304_222439"  # 13 16 19
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS7_loc20210304_222507"  # 13 13 17
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS9_loc20210304_222512"  # 10 13 17
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS11_loc20210304_222519"  # 11 15 19
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS13_loc20210304_222525"  # 13 14 21
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS15_loc20210304_222531"  # 12 13 21
folder_b = "0304_biomarker_df_dataset_iPAGE_coco_nc2020_seed69_dataRS17_loc20210304_222535"  # 11 15 23
# folder_b = ""  #
folder_gene_compare(folder_a, folder_b)


