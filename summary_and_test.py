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
# device = torch.device("cpu")  # TODO cpu


num_classes = 3


def test_lda_RF(lda_path, RF_path, data, label, result_final_save_path, model_name):  # ldaRF
    print(model_name)
    lda = pickle.load(open(lda_path, "rb"))
    clf_RF = pickle.load(open(RF_path, "rb"))
    X_test_after_LDA = lda.transform(data)
    y_pred = clf_RF.predict(X_test_after_LDA)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_onehot, result_final_save_path, model_name=model_name)

    return AUC_0, AUC_1, AUC_2, y_pred, y_pred_onehot


def test_sklearn(clf_path, data, label, result_final_save_path, model_name):  # CART SVM
    clf = pickle.load(open(clf_path, "rb"))
    y_pred = clf.predict(data)
    # print(y_pred)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    # print(y_pred)
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_onehot, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred, y_pred_onehot

def test_pytorch(nn_path, data, label, result_final_save_path, model_name):
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    data_tc, lable_tc = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)
    model = torch.load(nn_path)  # TODO cpu
    model.eval()
    y_pred = model(data_tc)
    # print(y_pred)
    y_pred_numpy = y_pred.cpu().detach().numpy()
    y_pred_numpy_out = y_pred_numpy.argmax(axis=1)

    # print(y_pred_numpy.argmax(axis=1))
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_numpy, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred_numpy_out, y_pred_numpy


