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

def load_data_raw(dataset="coconut", filter_nums=[3, 10]):
    """
    :param dataset: coconut coco_nc2020 test
    :param filter_nums:
    :return:
    """
    if dataset == "coconut":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "66099", "63990"]
        data_paths = ["data/coconut_20210127/"]
    elif dataset == "coco_nc2020":
        GSE_IDs = ["20346", "40012", "40396", "42026", "60244", "63990", "66099",
                   "21802", "27131", "28750", "42834", "57065", "68310", "69528", "111368"]
        data_paths = ["data/coconut_20210127/", "data/nc2020/"]
    elif dataset == "test":
        GSE_IDs = ["40012"]
        data_paths = ["data/coconut_20210127/"]
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
    label_GSE_concated = pd.concat(label_GSE, axis=0)
    return gene_GSE, label_GSE_concated


if __name__ == "__main__":
    gene_GSE, label_GSE_concated = load_data_raw()
    print(label_GSE_concated["label"].values)
    num_classes = len(pd.Categorical(label_GSE_concated["label"].values).categories)
    print(num_classes)