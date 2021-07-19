import pickle
from summary_and_train import multi_eval
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3


def test_lda_RF(lda_path, RF_path, data, label, result_final_save_path, model_name):  # ldaRF
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
    y_pred_onehot = np.eye(num_classes)[y_pred]
    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_onehot, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred, y_pred_onehot

def test_pytorch(nn_path, data, label, result_final_save_path, model_name):
    data_tc, lable_tc = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)
    model = torch.load(nn_path)
    model.eval()
    y_pred = model(data_tc)
    y_pred_numpy = y_pred.cpu().detach().numpy()
    y_pred_numpy_out = y_pred_numpy.argmax(axis=1)

    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_numpy, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred_numpy_out, y_pred_numpy


