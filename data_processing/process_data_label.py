import numpy as np
from data_processing.iPAGE import get_delta_with_fisher_exact_test


def get_data_with_ipage(gene_GSE_concated, label, threshold):
    delta_in_pair_pandas, pair_index_exact_expressed_list_final = get_delta_with_fisher_exact_test(
        gene_GSE_concated, label, threshold)
    delta_2level = (delta_in_pair_pandas <= 0) * (-1) + (delta_in_pair_pandas > 0) * 1
    data = delta_2level.values
    return data, pair_index_exact_expressed_list_final


def get_label_multilabel(label_GSE_concated):
    """
    :param label_GSE_concated:
    :return:
    """
    label = label_GSE_concated["label"].values
    label = label.astype(np.int16)
    return label


def get_label_0(label_GSE_concated):
    label = (label_GSE_concated["label"].values == 0) * 1
    return label


def get_label_1(label_GSE_concated):
    label = (label_GSE_concated["label"].values == 1) * 1
    return label


def get_label_2(label_GSE_concated):
    label = (label_GSE_concated["label"].values == 2) * 1
    return label


def get_label_3(label_GSE_concated):
    label = (label_GSE_concated["label"].values == 3) * 1
    return label
