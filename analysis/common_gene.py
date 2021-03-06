from utils import load_list_of_tuple
from utils import save_list_of_tuple
from utils import save_list_of_RNA_str
import glob
pair_list_glob = glob.glob("../results/20210304_common_gene/0304_biomarker_*/*/pair_after_lasso.csv")
pair_list_glob = sorted(pair_list_glob)
# print(pair_list_glob)
# acc_tuple = 0
for i, pair_glob in enumerate(pair_list_glob):
    print(pair_glob)
    if i == 0:
        acc_tuple = load_list_of_tuple(pair_glob)

    pair_tuple = load_list_of_tuple(pair_glob)
    acc_tuple_new = []
    for pair_tuple_each in pair_tuple:
        print(pair_tuple_each)
        rna_1, rna_2 = pair_tuple_each
        if (rna_1, rna_2) in acc_tuple or (rna_2, rna_1) in acc_tuple:
            acc_tuple_new.append(pair_tuple_each)
            # print(pair_tuple_each, "in", acc_tuple)
            print("in")
        else:
            print("no")
    print(len(acc_tuple_new))
    acc_tuple = acc_tuple_new
    # acc_tuple = acc_tuple&pair_tuple
    # print(acc_tuple)
    # input()
# save_list_of_tuple(acc_tuple, "../results/20210304_common_gene/common_gene.csv")
save_list_of_RNA_str(acc_tuple, "../results/20210304_common_gene/common_gene.csv")