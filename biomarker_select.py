from data_processing.process_data_label import get_data_with_ipage
from data_processing.process_data_label import get_label_0
from data_processing.process_data_label import get_label_1
from data_processing.process_data_label import get_label_2
from data_processing.process_data_label import get_label_3
from sklearn import metrics
import pickle
import os
import sklearn
from sklearn import linear_model
import numpy as np
from data_processing.iPAGE import calculate_delta_and_relative_expression

def binary_eval(y_test, y_pred, result_save_path, model_name=None):
    AUC = -1
    try:
        AUC = metrics.roc_auc_score(y_test, y_pred)
    except ValueError:
        print("one class wrong!")
    # AUC = metrics.roc_auc_score(y_test, y_pred)
    print(model_name + " AUC: " + str(AUC))
    f = open(result_save_path, "a+")
    f.write(model_name + "," + str(AUC) + ",")
    f.close()

def train_eval(X_train, X_test, y_train, y_test, result_save_path, SEED=None):
    print(f"SEED: {SEED} in LASSO")
    clf = sklearn.linear_model.Lasso(alpha=0.1, random_state=SEED)  # random_state 不影响筛选出的基因
    clf.fit(X_train, y_train)
    lasso_coef_not_zeros_index = np.nonzero(clf.coef_)[0]
    y_pred = clf.predict(X_test)
    model_name = f"Lasso of SEED{SEED}"
    binary_eval(y_test, y_pred, result_save_path, model_name)
    lasso_coef_pair_index = lasso_coef_not_zeros_index
    return clf.coef_, lasso_coef_pair_index


def biomarker_select(gene_GSE_concated_train, gene_GSE_concated_test, label_GSE_concated_train, label_GSE_concated_test, label_selected="control", path_data_prepared="data_prepared/test/", result_path = "results/test/2categories/", local_time=0, SEED=None):
    """
    :param label_selected:  select from "control" "bacteria" "virus" "coinfected"
    :param path_data_prepared:
    :param result_path:
    :param local_time:
    :param sklearn_random:
    :param prepared_data_func_str: "get_data_with_ipage" "get_data_ipage_4level"
    :return:
    """
    # parameters
    if not os.path.exists(path_data_prepared + label_selected + "/"):
        os.makedirs(path_data_prepared + label_selected + "/")

    if label_selected == "control":
        get_label_func = get_label_0
    elif label_selected == "bacteria":
        get_label_func = get_label_1
    elif label_selected == "virus":
        get_label_func = get_label_2
    elif label_selected == "coinfected":
        get_label_func = get_label_3
    else:
        print("label function is not found")
        exit()
        return

    prepared_data_path = path_data_prepared + label_selected + "/data_prepared_with_iPAGE.pickle"

    # label
    label_train = get_label_func(label_GSE_concated=label_GSE_concated_train)
    label_test = get_label_func(label_GSE_concated=label_GSE_concated_test)
    # data
    if os.path.exists(prepared_data_path):
        data_train, pair_index_exact_expressed_list_final = pickle.load(open(prepared_data_path, "rb"))
    else:
        data_train, pair_index_exact_expressed_list_final = get_data_with_ipage(gene_GSE_concated_train, label_train)
        pickle.dump((data_train, pair_index_exact_expressed_list_final), open(prepared_data_path, "wb"))
    data_test = calculate_delta_and_relative_expression(pair_index_exact_expressed_list_final, gene_GSE_concated_test)

    # LASSO select
    _, lasso_coef_pair_index = train_eval(data_train, data_test, label_train, label_test, result_save_path=result_path+f"result_2_categories_{local_time}.csv", SEED=SEED)

    return lasso_coef_pair_index, pair_index_exact_expressed_list_final
