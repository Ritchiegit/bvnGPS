import numpy as np
import pandas as pd
import glob as glob


def filter_label(gene_one_GSE, label, filter_num=3):
    label_still_index = np.nonzero(label["label"].values != filter_num)[0]
    if label_still_index.shape[0] != label.shape[0]:
        print("label_still_index.shape[0], label.shape[0]", label_still_index.shape[0], label.shape[0])
    label_still = label.iloc[label_still_index]
    gene_one_GSE_still = gene_one_GSE.iloc[:, label_still_index]
    return gene_one_GSE_still, label_still


def load_data_raw(dataset="coconut", filter_nums=None):
    """
    :param dataset: coconut coco_nc2020 test
    :param filter_nums:
    :return:
    """
    if filter_nums is None:
        filter_nums = [3, 10]
    if dataset == "coconut":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "66099", "63990"]  # 顺序需要注意
        data_paths = ["data/coconut_20210127/"]
    elif dataset == "coco_nc2020":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                   "21802", "27131", "28750", "42834", "57065", "68310", "69528", "111368"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/"]
    elif dataset == "test":
        GSE_IDs = ["40012"]
        data_paths = ["data/coconut_20210127/"]
    elif dataset == "GSE6269":
        # GSE_IDs = ["6269_1"]  # , "6269_2", "6269_3"]
        GSE_IDs = ["6269_2", "6269_3"]  # , "", "6269_3"]
        data_paths = ["data/host2016_20210322/"]
    elif dataset == "all_exclude_21802_57065":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                            "27131", "28750", "42834",          "68310", "69528", "111368",
                   "6269_2", "6269_3"]
                   # "6269_1", "6269_2", "6269_3"]  # 如果使用 GPL2507 处理出来的数据集 共有基因个数只有4794个 可能很影响性能。
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "only_21802_57065":
        GSE_IDs = ["21802", "57065"]
        data_paths = ["data/nc2020/"]
    elif dataset == "COVID19":
        GSE_IDs = ["152418"] #, "157859", "166253"]
        data_paths = ["data/COVID19/"]

    elif dataset == "all_exclude_57065":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                   "21802", "27131", "28750", "42834",          "68310", "69528", "111368",
                   "6269_2", "6269_3"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "all_exclude_21802":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                            "27131", "28750", "42834", "57065", "68310", "69528", "111368",
                   "6269_2", "6269_3"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "coco_nc2020_host":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                   "21802", "27131", "28750", "42834", "57065", "68310", "69528", "111368",
                   "6269_2", "6269_3"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "coco_nc2020_exclude_57065":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",  # 顺序需要注意
                   "21802", "27131", "28750", "42834",          "68310", "69528", "111368",
                   ]
        data_paths = ["data/coconut_20210127/", "data/nc2020/"]

    elif dataset == "only_21802":
        GSE_IDs = ["21802"]
        data_paths = ["data/nc2020/"]
    elif dataset == "only_57065":
        GSE_IDs = ["57065"]
        data_paths = ["data/nc2020/"]
    else:
        print("请检查数据集字符串格式")
        input()
        return
    gene_GSE = []
    label_GSE = []
    sample_sum = 0
    for GSE_ID in GSE_IDs:
        gene_need_to_concatenate_list = []

        for data_path in data_paths:
            mRNA_glob = glob.glob(data_path + f"exp_gene_GSE{GSE_ID}.txt")
            print(mRNA_glob)
            if len(mRNA_glob) != 0:
                break
        keys = []
        if len(mRNA_glob) != 0:
            mRNA = pd.read_csv(mRNA_glob[0], sep="\t")
            gene_need_to_concatenate_list.append(mRNA)
            keys.append("mRNA")
        gene_one_GSE = pd.concat(gene_need_to_concatenate_list, axis=0, keys=keys)
        print(gene_one_GSE.shape)

        for data_path in data_paths:
            label_glob = glob.glob(data_path + f"label_GSE{GSE_ID}.txt")
            print(label_glob)
            if len(label_glob) != 0:
                break
        label = pd.read_csv(label_glob[0], sep='\t', names=["id", "details", "label"])
        label = label.drop(index=[0])
        for filter_num in filter_nums:
            gene_one_GSE, label = filter_label(gene_one_GSE, label, filter_num)
        sample_sum += gene_one_GSE.shape[1]
        gene_GSE.append(gene_one_GSE)  # pd
        label_GSE.append(label)  # pd
        print("***" * 10)
    print(sample_sum)
    # label_GSE_concated = pd.concat(label_GSE, axis=0)
    # return gene_GSE, label_GSE_concated
    return gene_GSE, label_GSE


if __name__ == "__main__":
    import os
    os.chdir("..")
    # gene_GSE, label_GSE = load_data_raw(dataset="only_21802_57065")
    gene_GSE, label_GSE = load_data_raw(dataset="all_exclude_21802_57065")
    # gene_GSE, label_GSE = load_data_raw(dataset="COVID19")
    # gene_GSE, label_GSE = load_data_raw(dataset="GSE6269")
    label_GSE_concated = pd.concat(label_GSE, axis=0)
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
    gene_GSE_concated = gene_GSE_concated.T
    # for gene_GSE_one in gene_GSE:
        # print(gene_GSE_one)
    print(label_GSE_concated.shape)
    print(gene_GSE_concated.shape)
    num_classes = len(pd.Categorical(label_GSE_concated["label"].values).categories)
    print(num_classes)
    from data_processing.process_data_label import get_label_multilabel
    label = get_label_multilabel(label_GSE_concated)
    def sumall(label, zhi):
        return sum(label == zhi)
    a = sumall(label, 0)
    b = sumall(label, 1)
    c = sumall(label, 2)
    print(a, b, c)