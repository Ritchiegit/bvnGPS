from sklearn import metrics
import numpy as np
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from model.neural_network_0719 import NN_2layer_train_test, NN_2layer_subnet_train_test, NN_2layer_subnet_seperate_train_test
from model.mmoe import Config
from sklearn import svm
import pickle
import os
def multi_eval(y_test, y_pred, result_save_path=None, model_name=None, end_epoch=0):
    AUC_0, AUC_1, AUC_2 = -1, -1, -1
    # 按照最大值所属类别
    # y_pred_index = np.argmax(y_pred, axis=1)
    # try:
    #     AUC_0 = metrics.roc_auc_score(1 * (y_test == 0), 1 * (y_pred_index == 0))
    # except ValueError:
    #     print("uninfected wrong!")
    # try:
    #     AUC_1 = metrics.roc_auc_score(1*(y_test == 1), 1*(y_pred_index == 1))
    # except ValueError:
    #     print("bacteria wrong!")
    # try:
    #     AUC_2 = metrics.roc_auc_score(1*(y_test == 2), 1*(y_pred_index == 2))
    # except ValueError:
    #     print("virus wrong!")

    # 按照输出的置信度
    try:
        AUC_0 = metrics.roc_auc_score(1 * (y_test == 0), y_pred[:, 0])
    except ValueError:
        print("uninfected wrong!")
    try:
        AUC_1 = metrics.roc_auc_score(1*(y_test == 1), y_pred[:, 1])
    except ValueError:
        print("bacteria wrong!")
    try:
        AUC_2 = metrics.roc_auc_score(1*(y_test == 2), y_pred[:, 2])
    except ValueError:
        print("virus wrong!")
    print(f"{model_name},Health{AUC_0},Bacteria{AUC_1},Virus{AUC_2},End_epoch{end_epoch}\n")
    if result_save_path is not None:
        f = open(result_save_path, "a+")
        f.write(f"{model_name},{AUC_0},{AUC_1},{AUC_2},{end_epoch}\n")
    AUC_mean = sum((AUC_0, AUC_1, AUC_2))/3
    return AUC_0, AUC_1, AUC_2, AUC_mean

def summary_and_train(train_data_all, test_data_all, label_train, label_test, result_save_path="results/test/", local_time=0, sklearn_random=109, config={}):

    """
    train with all selected biomarkers
    :param train_data_all:
    :param test_data_all:
    :param label_train:
    :param label_test:
    :param result_full_filepath:
    :param sklearn_random:
    :return:
    """
    model_save_folder_path = result_save_path + "_model/"
    if not os.path.exists(model_save_folder_path):
        os.makedirs(model_save_folder_path)

    result_full_filepath = result_save_path + f"result_summary_{local_time}.csv"
    f = open(result_full_filepath, "a+")
    f.close()
    X_train, X_test, y_train, y_test = train_data_all, test_data_all, label_train, label_test

    X_train = np.concatenate(train_data_all, axis=1)
    X_test = np.concatenate(test_data_all, axis=1)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    num_classes = len(pd.Categorical(label_train).categories)

    ############################################### subnet三分类 #################################################
    optimizer_str = 'Adam'
    # hidden_feature = 16
    learning_rate = 0.001
    # config['hidden_feature'] = hidden_feature
    # config['classifier_feature_list'] = [32]
    # print(config['classifier_feature_list'])
    # y_pred, end_epoch, model_name = NN_2layer_subnet_seperate_train_test(X_train, X_test, y_train, y_test, num_classes, 2000,
    #                                                      sklearn_random=sklearn_random, criterion_type="MSE",
    #                                                      learning_rate=learning_rate, optimizer_str=optimizer_str,
    #                                                      model_save_folder_path=model_save_folder_path, config=config,
    #                                                      model_str=config['model_str'])
    # multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    learning_rates = [0.001]
    optimizer_str_list = ["Adam"]
    hidden_feature_to_search = [2, 4, 8, 16, 32, 64]  # 76
    for optimizer_str in optimizer_str_list:
        for learning_rate in learning_rates:
            for hidden_feature in hidden_feature_to_search:
                config['hidden_feature'] = hidden_feature
                config['classifier_feature_list'] = []
                print(config['classifier_feature_list'])
                y_pred, end_epoch, model_name = NN_2layer_subnet_seperate_train_test(X_train, X_test, y_train, y_test,
                                                                            num_classes,
                                                                            2000, sklearn_random=sklearn_random,
                                                                            criterion_type="MSE",
                                                                            learning_rate=learning_rate,
                                                                            optimizer_str=optimizer_str,
                                                                            model_save_folder_path=model_save_folder_path,
                                                                            config=config,
                                                                            model_str=config['model_str'])
                multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    learning_rates = [0.001]
    optimizer_str_list = ["Adam"]
    hidden_feature_to_search = [2, 4, 8, 16, 32, 64, 128]  # 76
    layer1_list = [2, 4, 8, 16, 32, 64, 128]
    for optimizer_str in optimizer_str_list:
        for learning_rate in learning_rates:
            for hidden_feature in hidden_feature_to_search:
                for layer1 in layer1_list:
                    # if layer1 > hidden_feature: break
                    config['hidden_feature'] = hidden_feature
                    config['classifier_feature_list'] = [layer1]

                    print(config['classifier_feature_list'])
                    y_pred, end_epoch, model_name = NN_2layer_subnet_seperate_train_test(X_train, X_test, y_train, y_test,
                                                                                num_classes,
                                                                                2000, sklearn_random=sklearn_random,
                                                                                criterion_type="MSE",
                                                                                learning_rate=learning_rate,
                                                                                optimizer_str=optimizer_str,
                                                                                model_save_folder_path=model_save_folder_path,
                                                                                config=config,
                                                                                model_str=config['model_str'])
                    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    learning_rates = [0.001]
    optimizer_str_list = ["Adam"]
    hidden_feature_to_search = [2, 4, 8, 16, 32, 64, 128]  # 76
    layer1_list = [2, 4, 8, 16, 32, 64, 128]
    layer2_list = [2, 4, 8, 16, 32, 64, 128]
    for optimizer_str in optimizer_str_list:
        for learning_rate in learning_rates:
            for hidden_feature in hidden_feature_to_search:
                for layer1 in layer1_list:
                    # if layer1 > hidden_feature: break
                    for layer2 in layer2_list:
                        if layer2 > layer1: break
                        config['hidden_feature'] = hidden_feature
                        config['classifier_feature_list'] = [layer1, layer2]

                        print(config['classifier_feature_list'])
                        y_pred, end_epoch, model_name = NN_2layer_subnet_seperate_train_test(X_train, X_test, y_train, y_test,
                                                                                    num_classes, 2000,
                                                                                    sklearn_random=sklearn_random,
                                                                                    criterion_type="MSE",
                                                                                    learning_rate=learning_rate,
                                                                                    optimizer_str=optimizer_str,
                                                                                    model_save_folder_path=model_save_folder_path,
                                                                                    config=config,
                                                                                    model_str=config['model_str'])
                        multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)


    # CART
    model_name = "CART"
    print(model_name)
    clf_cart = tree.DecisionTreeClassifier()
    clf_cart = clf_cart.fit(X_train, y_train)
    y_pred = clf_cart.predict(X_test)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)
    model_save_file_path = model_save_folder_path + model_name + ".model_pickle"
    pickle.dump(clf_cart, open(model_save_file_path, "wb"))

    # LDA + RF
    model_name = "RandomForest"
    print(model_name)
    lda = LDA(n_components=1)
    X_train_after_LDA = lda.fit_transform(X_train, y_train)
    X_test_after_LDA = lda.transform(X_test)
    clf_RF = RandomForestClassifier()
    # clf_RF = RandomForestClassifier(random_state=108)
    clf_RF.fit(X_train_after_LDA, y_train)
    y_pred = clf_RF.predict(X_test_after_LDA)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)
    model_save_file_path = model_save_folder_path + model_name + ".model_pickle"
    model_save_file_lda_path = model_save_folder_path + "lda" + ".model_pickle"
    pickle.dump(lda, open(model_save_file_lda_path, "wb"))
    pickle.dump(clf_RF, open(model_save_file_path, "wb"))

    # SVM with one vs. one  40
    kernel_list = ["linear"]
    decision_function_shape_list = ["ovo", "ovr"]
    # decision_function_shape_list = ["ovo"]
    C_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # C_list = [0.2]
    for kernel in kernel_list:
        for decision_function_shape in decision_function_shape_list:
            for C in C_list:
                model_name = f"SVM_k{kernel}_d{decision_function_shape}_C{C}"
                print(model_name)
                clf_svm = svm.SVC(C=C, kernel=kernel, decision_function_shape=decision_function_shape)
                clf_svm.fit(X_train, y_train)
                y_pred = clf_svm.predict(X_test)
                y_pred_onehot = np.eye(num_classes)[y_pred]
                multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)

                model_save_file_path = model_save_folder_path + model_name + ".model_pickle"
                pickle.dump(clf_svm, open(model_save_file_path, "wb"))
