import glob
import pickle
# from summary_and_train import multi_eval
from multi_eval import multi_eval
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


def test_pytorch_three_input_model(nn_path, data, label, result_final_save_path, model_name):
    print("data", data[0].shape)
    print("data", data[1].shape)
    print("data", data[2].shape)
    data = np.concatenate(data, axis=1)  # todo
    print("data concatenate test_model 0719", data.shape)

    data_tc, lable_tc = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)
    model = torch.load(nn_path)  # TODO cpu
    model.eval()
    y_pred = model(data_tc)
    y_pred_numpy = y_pred.cpu().detach().numpy()
    y_pred_numpy_out = y_pred_numpy.argmax(axis=1)

    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_numpy, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred_numpy_out, y_pred_numpy

def test_pytorch_three_input_model_subnet(nn_path, data, label, result_final_save_path, model_name):
    print("data", data[0].shape)
    print("data", data[1].shape)
    print("data", data[2].shape)
    data = np.concatenate(data, axis=1)  # todo
    print("data concatenate test_model 0719", data.shape)

    data_tc, lable_tc = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)
    model = torch.load(nn_path)  # TODO cpu
    model.eval()
    y_pred = model(data_tc)
    y_pred_numpy = y_pred[0].cpu().detach().numpy()
    y_pred_numpy_out = y_pred_numpy.argmax(axis=1)

    AUC_0, AUC_1, AUC_2, AUC_mean = multi_eval(label, y_pred_numpy, result_final_save_path, model_name=model_name)
    return AUC_0, AUC_1, AUC_2, y_pred_numpy_out, y_pred_numpy


