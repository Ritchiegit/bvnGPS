
import pandas as pd
import os
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
os.chdir("..")

dataset = "only_21802_57065"
type_part_dataset = None
dataset = "all_exclude_21802_57065"
type_part_dataset = "0.3"
dataset = "all_exclude_21802_57065"
type_part_dataset = "0.7"
label_file = f"{dataset}{type_part_dataset}.csv"
dataset_random_state = 1

gene_GSE, label_GSE = load_data_raw(dataset=dataset)
label_GSE_concated = pd.concat(label_GSE, axis=0)
gene_GSE_concated = pd.concat(gene_GSE, join="inner", axis=1)
gene_GSE_concated = gene_GSE_concated.T
if type_part_dataset == "0.7":
    gene_GSE_concated, _, label_GSE_concated, _ = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    # print("type 0.7")
    # print(gene_GSE_concated.shape)
elif type_part_dataset == "0.3":
    _, gene_GSE_concated, _, label_GSE_concated = train_test_split(
        gene_GSE_concated, label_GSE_concated, test_size=0.3, random_state=dataset_random_state)
    print("type 0.3")
    print(gene_GSE_concated.shape)
else:
    assert type_part_dataset is None

label = get_label_multilabel(label_GSE_concated=label_GSE_concated)
print(label.shape)
label_pd = pd.DataFrame(label)
label_pd.to_csv(label_file, index=False, header=False)

