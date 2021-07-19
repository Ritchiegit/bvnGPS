import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm


def comb_mine(N, k):
    up = np.math.factorial(N)
    down = (np.math.factorial(k) * np.math.factorial(N - k))
    vals = up // down
    return vals


def fisher_exact_test(gene_GSE_T, label, threshold=1e-16):
    data = gene_GSE_T.values
    columns = gene_GSE_T.columns
    len_gene = len(gene_GSE_T.columns)
    pair_index_exact_expressed = []
    pair_index_exact = []
    max_num = len_gene * (len_gene - 1) / 2
    # max_num = 0
    for i in range(0, len_gene - 1):
        for j in range(i + 1, len_gene):
            pair_each = data[:, i] > data[:, j]
            a = np.sum((pair_each > 0.5) * (label < 0.5))
            c = np.sum((pair_each > 0.5) * (label > 0.5))
            b = np.sum((pair_each < 0.5) * (label < 0.5))
            d = np.sum((pair_each < 0.5) * (label > 0.5))
            n = a + b + c + d
            p = comb_mine(a + b, a) * comb_mine(c + d, c) / comb_mine(n, a + c)
            if p < threshold:
                pair_index_exact.append((i, j))
                pair_index_exact_expressed.append((columns[i], columns[j]))
            # max_num += 1
    return pair_index_exact, pair_index_exact_expressed, max_num


def load_pathway():
    pathway = "data/group_gene/pathway_label/c2.cp.kegg.v7.2.symbols.gmt"
    gene_in_pathways = []
    for line in open(pathway):
        line = line.strip()
        split_result = line.split('\t')
        gene_in_pathway = split_result[2:]
        gene_in_pathways.append(gene_in_pathway)
    return gene_in_pathways


def get_delta_with_fisher_exact_test(gene_GSE_adjusted_concated, label, threshold=1e-16):
    """
    :param gene_GSE_adjusted_concated: pandas data frame concated
    :param label:
    :return:
    """
    gene_in_pathways = load_pathway()
    col_RNA = gene_GSE_adjusted_concated.columns

    RNA_in_dataset = list(zip(*col_RNA))[1]
    RNA_in_dataset_set = set(RNA_in_dataset)
    common_RNA_list = []
    for gene_in_pathway in gene_in_pathways:

        gene_in_pathway_set = set(gene_in_pathway)
        common_RNA = list(gene_in_pathway_set & RNA_in_dataset_set)
        common_RNA = sorted(common_RNA)  # ensure that the same random number has the same result
        if len(common_RNA) == 0:
            continue
        common_RNA_list.append(common_RNA)
    gene_GSE_in_pathways = []
    for common_RNA in common_RNA_list:  # every one is corresponding to one pathway
        mRNA_aux = ["mRNA" for _ in range(len(common_RNA))]
        common_RNA_concat = list(zip(mRNA_aux, common_RNA))
        gene_GSE_in_pathway = gene_GSE_adjusted_concated[common_RNA_concat]  # 每个pathway中基因的表现型
        gene_GSE_in_pathways.append(gene_GSE_in_pathway)
    # Calculate the result of the pair in each pathway.
    pair_index_exact_expressed_list = []
    print("calculate fisher exact in pathway")
    sum_max_num = 0
    for gene_GSE_in_pathway in tqdm(gene_GSE_in_pathways):
        pair_index_exact, pair_index_exact_expressed, max_num = fisher_exact_test(gene_GSE_in_pathway, label, threshold)
        pair_index_exact_expressed_list.append(pair_index_exact_expressed)  # pair name
        sum_max_num += max_num
    print("after fisher exact test of all pathway")
    pair_index_exact_expressed_list_final = list(set(list(itertools.chain(*pair_index_exact_expressed_list))))
    # pair_index_exact_expressed_list_final = sorted(pair_index_exact_expressed_list_final)  # 可用于保证输出的pair次序相同，但本行代码不对最终结果产生影响。
    delta_in_pair_list = []
    for col_name_1, col_name_2 in tqdm(pair_index_exact_expressed_list_final):
        col1 = gene_GSE_adjusted_concated[col_name_1]
        col2 = gene_GSE_adjusted_concated[col_name_2]
        delta_in_pair = col1 - col2
        delta_in_pair_list.append(delta_in_pair)
    delta_in_pair_pandas = pd.concat(delta_in_pair_list, axis=1)
    print("len(delta_in_pair_list)", len(delta_in_pair_list))
    print("sum_max_num", sum_max_num)
    return delta_in_pair_pandas, pair_index_exact_expressed_list_final


def calculate_delta_and_relative_expression(pair_index_exact_expressed_list_final, gene_GSE_concated):
    delta_in_pair_list = []
    for col_name_1, col_name_2 in pair_index_exact_expressed_list_final:
        col1 = gene_GSE_concated[col_name_1]
        col2 = gene_GSE_concated[col_name_2]
        delta_in_pair = col1 - col2
        delta_in_pair_list.append(delta_in_pair)
    delta_in_pair_pandas = pd.concat(delta_in_pair_list, axis=1).values
    delta_after_quantization = (delta_in_pair_pandas <= 0) * (-1) + (delta_in_pair_pandas > 0) * (1)

    return delta_after_quantization
