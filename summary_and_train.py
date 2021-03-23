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
from model.neural_network import NN_2layer_train_test
from model.mmoe import Config
from sklearn import svm
import pickle
import os
def multi_eval(y_test, y_pred, result_save_path=None, model_name=None, end_epoch=0):
    y_pred_index = np.argmax(y_pred, axis=1)
    AUC_0, AUC_1, AUC_2 = -1, -1, -1
    try:
        AUC_0 = metrics.roc_auc_score(1 * (y_test == 0), 1 * (y_pred_index == 0))
    except ValueError:
        print("uninfected wrong!")
    try:
        AUC_1 = metrics.roc_auc_score(1*(y_test == 1), 1*(y_pred_index == 1))
    except ValueError:
        print("bacteria wrong!")
    try:
        AUC_2 = metrics.roc_auc_score(1*(y_test == 2), 1*(y_pred_index == 2))
    except ValueError:
        print("virus wrong!")
    # AUC_0 = metrics.roc_auc_score(1*(y_test == 0), 1*(y_pred_index == 0))
    # AUC_1 = metrics.roc_auc_score(1*(y_test == 1), 1*(y_pred_index == 1))
    # AUC_2 = metrics.roc_auc_score(1*(y_test == 2), 1*(y_pred_index == 2))
    #  print(f"{model_name},Health{AUC_0},Bacteria{AUC_1},Virus{AUC_2}\n")
    print(f"{model_name},Health{AUC_0},Bacteria{AUC_1},Virus{AUC_2},End_epoch{end_epoch}\n")
    if result_save_path is not None:
        f = open(result_save_path, "a+")
        f.write(f"{model_name},{AUC_0},{AUC_1},{AUC_2},{end_epoch}\n")
    AUC_mean = sum((AUC_0, AUC_1, AUC_2))/3
    return AUC_0, AUC_1, AUC_2, AUC_mean

def summary_and_train(train_data_all, test_data_all, label_train, label_test, result_save_path="results/test/", local_time=0, sklearn_random=109):
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
    num_classes = len(pd.Categorical(label_train).categories)

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
    clf_RF = RandomForestClassifier(random_state=108)
    clf_RF.fit(X_train_after_LDA, y_train)
    y_pred = clf_RF.predict(X_test_after_LDA)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)
    model_save_file_path = model_save_folder_path + model_name + ".model_pickle"
    model_save_file_lda_path = model_save_folder_path + "lda" + ".model_pickle"
    pickle.dump(lda, open(model_save_file_lda_path, "wb"))
    pickle.dump(clf_RF, open(model_save_file_path, "wb"))

    # SVM with one vs. one
    # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,  # ‘precomputed’ precomputed 需要预先计算kernel的值
    kernel_list = ["linear", "poly", "rbf", "sigmoid"]
    decision_function_shape_list = ["ovo", "ovr"]
    C_list = [1]
    C_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
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
    # SVM with one vs. rest


    # Multi layer FCN
    """
    learning_rate = 0.001
    optimizer_str = "Adam"
    hidden_feature_list = [8, 4]
    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000, sklearn_random=sklearn_random,
                                             criterion_type="MSE", learning_rate=learning_rate, optimizer_str=optimizer_str,
                                             model_str="multi_layers_FCN", hidden_feature_list=hidden_feature_list, model_save_folder_path=model_save_folder_path)
    model_name = f"MFCN_MSE_opt{optimizer_str}_h{hidden_feature_list}_lr{learning_rate}"
    print(model_name)
    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    """

    learning_rate = 0.001
    optimizer_str_list = ["Adam", "Adagrad"]
    layer_1_num_list = [2, 8, 32, 128, 512]
    layer_2_num_list = [2, 8, 32, 128, 512]
    for optimizer_str in optimizer_str_list:
        for layer_1_num in layer_1_num_list:
            for layer_2_num in layer_2_num_list:
                hidden_feature_list = [layer_1_num, layer_2_num]
                model_name = f"MFCN_MSE_opt{optimizer_str}_h{layer_1_num}_h{layer_2_num}_lr{learning_rate}"
                print(model_name)  # TODO 没有将这个model name 传进去
                y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000, sklearn_random=sklearn_random,
                                                         criterion_type="MSE", learning_rate=learning_rate, optimizer_str=optimizer_str,
                                                         model_str="multi_layers_FCN", hidden_feature_list=hidden_feature_list, model_save_folder_path=model_save_folder_path, model_name_for_save=model_name)
                multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)


    # MMoE
    optimizer_str_list = ["Adam", "Adagrad"]
    learning_rate = 0.001
    num_experts_list = [16, 24, 32]  # [2, 4, 8, 12]
    # num_experts_list = [2, 4, 8, 12]
    expert_unit_list = [4, 8, 12, 16, 32]  # expert units
    hidden_units_list = [8, 16, 32]  # tower units
    hidden_feature = ""
    for optimizer_str in optimizer_str_list:
        for num_experts in num_experts_list:
            for expert_unit in expert_unit_list:
                for hidden_units in hidden_units_list:
                    config = Config(num_experts=num_experts, expert_unit=expert_unit, hidden_units=hidden_units)
                    model_name = f"MMoE_MSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}_nexp{num_experts}_uexp{expert_unit}_utower{hidden_units}"
                    print(model_name)
                    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000,
                                                  sklearn_random=sklearn_random, criterion_type="MSE",
                                                  hidden_feature=hidden_feature, learning_rate=learning_rate,
                                                  optimizer_str=optimizer_str, model_str="MMoE", config=config, model_save_folder_path=model_save_folder_path)
                    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    # hidden_feature = 4
    # optimizer_str = "Adam"
    # learning_rate = 0.001
    # model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
    # print(model_name)
    # y_pred = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000,
    #                               sklearn_random=sklearn_random, criterion_type="MSE",
    #                               hidden_feature=hidden_feature, learning_rate=learning_rate,
    #                               optimizer_str=optimizer_str, model_str="FCN", model_save_folder_path=model_save_folder_path)
    # multi_eval(y_test, y_pred, result_full_filepath, model_name)

    """
    hidden_feature = 4
    optimizer_str = "Adam"
    learning_rate = 0.001
    model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
    print(model_name)
    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000,
                                  sklearn_random=sklearn_random, criterion_type="MSE",
                                  hidden_feature=hidden_feature, learning_rate=learning_rate,
                                  optimizer_str=optimizer_str, model_str="MoE", model_save_folder_path=model_save_folder_path)
    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    """
    hidden_feature_to_search = [2, 4, 16, 64, 128, 256, 512, 600, 750, 800]
    # hidden_feature_to_search = [2,8, 32, 128, 256, 500, 1000]
    learning_rates = [0.001, 0.0005]
    optimizer_str_list = ["Adam", "Adagrad"]
    for optimizer_str in optimizer_str_list:
        for hidden_feature in hidden_feature_to_search:
            # Neural Network
            # hidden feature batch_size learning_rate (optimizer)
            for learning_rate in learning_rates:
                model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
                print(model_name)
                y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000, sklearn_random=sklearn_random, criterion_type="MSE", hidden_feature=hidden_feature, learning_rate=learning_rate, optimizer_str=optimizer_str, model_save_folder_path=model_save_folder_path)
                multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    # optimizer_str = "Adagrad"
    # for max_epoch in [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 35, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000]:
    #     # Neural Network
    #     hidden_feature = 96
    #     learning_rate = 0.001
    #     model_name = f"Neural Network MSE epoch{max_epoch}_opt{optimizer_str}"
    #     print(model_name)
    #     y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, max_epoch, sklearn_random=sklearn_random, criterion_type="MSE", hidden_feature=hidden_feature, learning_rate=learning_rate,
    #                                   earlystop_turn_on=False, val_ratio=0, optimizer_str=optimizer_str, model_save_folder_path=model_save_folder_path)
    #     multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    # # lgb multi-class
    # model_name = "lightgbm"
    # print(model_name)
    # gbm = lgb.LGBMRegressor(objective='multiclass', num_leaves=31, learning_rate=0.05, num_classes=num_classes)
    # gbm.fit(X_train, y_train)
    # y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # multi_eval(y_test, y_pred, result_full_filepath, model_name)

    f = open(result_full_filepath, "a+")
    f.write("\n")
    f.close()
