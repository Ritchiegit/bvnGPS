import pandas as pd
import numpy as np


def check_front_of_name(name, front):
    return name[:len(front)] == front


def check_end_of_name(name, end):
    return name[-len(end):] == end


def save_list_of_tuple(list_of_tuple, file_name_to_save="test_save_tuple.csv"):
    list_of_str = list(map(str, list_of_tuple))
    list_of_str_pd = pd.DataFrame(list_of_str)
    list_of_str_pd.to_csv(file_name_to_save, index=False, header=False)
    return


def load_list_of_tuple(file_name_for_read):
    list_of_str_pd = pd.read_csv(file_name_for_read, index_col=None, header=None)
    list_of_str_numpy = list_of_str_pd.values.flatten()
    list_of_tuple = list(map(eval, list_of_str_numpy))
    return list_of_tuple


def tuple_pair_2_gene_pair(tuple):
    tuple1, tuple2 = tuple
    print(tuple1, tuple2)
    return (tuple1[1], tuple2[1])


def save_list_of_RNA_str(list_of_tuple, file_name_to_save):
    list_of_str = list(map(tuple_pair_2_gene_pair, list_of_tuple))
    list_of_str_pd = pd.DataFrame(list_of_str)
    list_of_str_pd.to_csv(file_name_to_save, index=False, header=False)
    return


def list_with_index(list_in_list, index_in_array):
    new_list = []
    for index_in_array_each in index_in_array:
        new_list.append(list_in_list[index_in_array_each])
    return new_list


def select_common_gene_expression_from_train_test(gene_GSE_concated_train, gene_GSE_concated_test):
    gene_feature_idx_list = []
    for gene in [gene_GSE_concated_train, gene_GSE_concated_test]:
        gene_feature_idx_list.append(gene.columns)
    from functools import reduce
    common_gene_feature_idx = reduce(np.intersect1d, gene_feature_idx_list)
    gene_GSE_concated_train_new = gene_GSE_concated_train[common_gene_feature_idx]
    gene_GSE_concated_test_new = gene_GSE_concated_test[common_gene_feature_idx]
    return gene_GSE_concated_train_new, gene_GSE_concated_test_new


if __name__ == "__main__":
    c = [("a", "qe"), ("a", "c"), ("a", "d"), ("a", "e")]
    save_list_of_tuple(c, "0228.csv")
    b = load_list_of_tuple("0228.csv")
    print(c==b)
