import glob
import pickle
from summary_and_train import multi_eval
from model.FCN import FCN
from model.MoE import MoE
from model.mmoe import MMoE
from model.mmoe import Config
from model.multi_layers_FCN import mutli_layers_FCN
import torch
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3


def test_lda_RF(lda_path, RF_path, data, label, result_final_save_path, model_name):  # ldaRF
    print(model_name)
    lda = pickle.load(open(lda_path, "rb"))
    clf_RF = pickle.load(open(RF_path, "rb"))
    X_test_after_LDA = lda.transform(data)
    y_pred = clf_RF.predict(X_test_after_LDA)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_onehot, result_final_save_path, model_name=model_name)

    return AUC_0, AUC_1, AUC_2, AUC_mean


def test_sklearn(clf_path, data, label, result_final_save_path, model_name):  # CART SVM
    clf = pickle.load(open(clf_path, "rb"))
    y_pred = clf.predict(data)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    print(y_pred)
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_onehot, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, AUC_mean

def test_pytorch(nn_path, data, label, result_final_save_path, model_name):
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    data_tc, lable_tc = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)
    model = torch.load(nn_path)
    model.eval()
    y_pred = model(data_tc)
    print(y_pred)
    y_pred_numpy = y_pred.cpu().detach()
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_numpy, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, AUC_mean

def match_front_in_list(name_list, front):
    new_name_list = []
    len_front = len(front)
    for name in name_list:
        if name[:len_front] == front:
            new_name_list.append(name)
    return new_name_list

def summary_and_test(data, label, all_model_path, result_final_save_path, local_time=0, sklearn_random=1):
    print(data.shape)
    print(label.shape)
    print(result_final_save_path)
    print(local_time)
    print(sklearn_random)

    f = open(result_final_save_path, "w")
    f.close()

    name_list = os.listdir(all_model_path)
    name_list_CART = match_front_in_list(name_list, "CART")
    # name_list_RandomForest = match_front_in_list(name_list, "RandomForest")
    name_list_SVM = match_front_in_list(name_list, "SVM")
    name_list_FCN = match_front_in_list(name_list, "FCN")
    name_list_MMoE = match_front_in_list(name_list, "MMoE")
    name_list_multi_layers_FCN = match_front_in_list(name_list, "multi_layers_FCN")
    print(name_list_CART)
    print(name_list_SVM)
    print(name_list_FCN)
    print(name_list_MMoE)
    print(name_list_multi_layers_FCN)

    # glob_pth = glob.glob()
    # results/final_model_results/test_1_iPAGE_coco_nc2020_seed157_dataRS1_model/

    # model_path_list = glob.glob(model_path + "*")
    # print(model_path_list)


    for model_name in name_list_CART:
        model_path = all_model_path + model_name
        a = test_sklearn(model_path, data, label, result_final_save_path, model_name=model_name)
        print(a)

    # lda
    a = test_lda_RF(all_model_path+"lda.model_pickle", all_model_path+"RandomForest.model_pickle", data, label, result_final_save_path,
                    model_name="RandomForest")
    print(a)


    for model_name in name_list_SVM:
        model_path = all_model_path + model_name
        a = test_sklearn(model_path, data, label, result_final_save_path, model_name=model_name)
        print(a)

    for model_name in name_list_FCN:
        model_path = all_model_path + model_name
        a = test_pytorch(model_path, data, label, result_final_save_path, model_name=model_name)
        print(a)


    for model_name in name_list_MMoE:
        model_path = all_model_path + model_name
        a = test_pytorch(model_path, data, label, result_final_save_path, model_name=model_name)
        print(a)

    for model_name in name_list_multi_layers_FCN:
        model_path = all_model_path + model_name
        a = test_pytorch(model_path, data, label, result_final_save_path, model_name=model_name)
        print(a)



