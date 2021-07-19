import numpy as np
import pandas as pd
import glob as glob

def list_1_minus_list_2(list1, list2):
    new_list = []
    for elem in list1:
        if elem not in list2:
            new_list.append(elem)
    return new_list

def filter_label(gene_one_GSE, label, filter_num=3):
    label_still_index = np.nonzero(label["label"].values != filter_num)[0]
    if label_still_index.shape[0] != label.shape[0]:
        print("label_still_index.shape[0], label.shape[0]", label_still_index.shape[0], label.shape[0])
    label_still = label.iloc[label_still_index]
    gene_one_GSE_still = gene_one_GSE.iloc[:, label_still_index]
    return gene_one_GSE_still, label_still

def label_replace(label, from_num, to_num):
    label["label"].loc[label["label"].values == from_num] = to_num
    return label

def load_data_raw(dataset="coconut", external_val_set=[], filter_nums=None, from_num_to_num=[]):
    """
    :param dataset: coconut coco_nc2020 test
    :param filter_nums:
    :return:
    """
    if filter_nums is None:
        filter_nums = [3, 10]
    if isinstance(dataset, list) or isinstance(dataset, tuple):
        GSE_IDs = list(map(str, dataset))
        print(GSE_IDs)
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "all_exclude2_raw63990":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990_raw_label", "66099",
                            "27131", "28750", "42834",          "68310", "69528", "111368",
                   # "6269_2", "6269_3"]
                   "6269_1", "6269_2", "6269_3"]
        # If the data set processed by GPL2507 is used,
        # the total number of genes is only 4,794, which may affect performance.
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "all_exclude2_6269_1_raw63990":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990_raw_label", "66099",  # 顺序需要注意
                            "27131", "28750", "42834",          "68310", "69528", "111368",
                   # "6269_2", "6269_3"]
                   "6269_2", "6269_3"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/", "data/host2016_20210322/"]
    elif dataset == "only_21802_57065":
        GSE_IDs = ["21802", "57065"]
        data_paths = ["data/nc2020/"]
    else:
        print("Please check the data set string format")
        exit(1)
        return
    external_val_set = list(map(str, external_val_set))
    GSE_IDs = list_1_minus_list_2(GSE_IDs, external_val_set)
    print("external_val_set", external_val_set)
    print("GSE_IDs", GSE_IDs)
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
            len_label_old = label.shape[0]
            gene_one_GSE, label = filter_label(gene_one_GSE, label, filter_num)
            len_label_new = label.shape[0]
            if filter_num == 3 and len_label_new != len_label_old:
                print("GSE_before_and_after_filter_3:", GSE_ID, "old:", len_label_old, "new:", len_label_new)
                # input()
        if len(from_num_to_num) == 2:
            print("in replace")
            label = label_replace(label, from_num_to_num[0], from_num_to_num[1])
        sample_sum += gene_one_GSE.shape[1]
        gene_GSE.append(gene_one_GSE)  # each element is a pandas.Dataframe
        label_GSE.append(label)  # each element is a pandas.Dataframe
        print("***" * 10)
    print(sample_sum)
    return gene_GSE, label_GSE


if __name__ == "__main__":
    import os
    os.chdir("..")
    gene_GSE, label_GSE = load_data_raw(dataset="all_exclude2_6269_1_raw63990")
    gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1).T
    dataset_random_state = 1
    label_GSE_concated = pd.concat(label_GSE, axis=0)
    from sklearn.model_selection import train_test_split
    gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    print("label_GSE_concated_train.shape", label_GSE_concated_train.shape)
    print("label_GSE_concated_test.shape", label_GSE_concated_test.shape)

    print(gene_GSE_concated.shape)
    print(gene_GSE_concated_train.shape)
    print(gene_GSE_concated_test.shape)

