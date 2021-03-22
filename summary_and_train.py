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
def multi_eval(y_test, y_pred, result_save_path, model_name=None, end_epoch=0):
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
    print(f"Health{AUC_0},Bacteria{AUC_1},Virus{AUC_2},End_epoch{end_epoch}\n")
    f = open(result_save_path, "a+")
    f.write(f"{model_name},{AUC_0},{AUC_1},{AUC_2},{end_epoch}\n")

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
    result_full_filepath = result_save_path + f"result_summary_{local_time}.csv"
    f = open(result_full_filepath, "a+")
    f.close()
    X_train, X_test, y_train, y_test = train_data_all, test_data_all, label_train, label_test
    num_classes = len(pd.Categorical(label_train).categories)

    # CART
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    model_name = "CART"
    print(model_name)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)

    # LDA + RF
    lda = LDA(n_components=1)
    X_train_after_LDA = lda.fit_transform(X_train, y_train)
    X_test_after_LDA = lda.transform(X_test)
    classifier = RandomForestClassifier(random_state=108)
    classifier.fit(X_train_after_LDA, y_train)
    y_pred = classifier.predict(X_test_after_LDA)
    y_pred_onehot = np.eye(num_classes)[y_pred]
    model_name = "RandomForest"
    print(model_name)
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)

    # Multi layer FCN
    """
    learning_rate = 0.001
    optimizer_str = "Adam"
    hidden_feature_list = [8, 4]
    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000, sklearn_random=sklearn_random,
                                             criterion_type="MSE", learning_rate=learning_rate, optimizer_str=optimizer_str,
                                             model_str="multi_layers_FCN", hidden_feature_list=hidden_feature_list)
    model_name = f"MFCN_MSE_opt{optimizer_str}_h{hidden_feature_list}_lr{learning_rate}"
    print(model_name)
    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    """

    # learning_rate = 0.001
    # optimizer_str_list = ["Adam", "Adagrad"]
    # layer_1_num_list = [2, 8, 32, 128, 512, 1024]
    # layer_2_num_list = [2, 8, 32, 128, 512, 1024]
    # for optimizer_str in optimizer_str_list:
    #     for layer_1_num in layer_1_num_list:
    #         for layer_2_num in layer_2_num_list:
    #             hidden_feature_list = [layer_1_num, layer_2_num]
    #             y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000, sklearn_random=sklearn_random,
    #                                                      criterion_type="MSE", learning_rate=learning_rate, optimizer_str=optimizer_str,
    #                                                      model_str="multi_layers_FCN", hidden_feature_list=hidden_feature_list)
    #             model_name = f"MFCN_MSE_opt{optimizer_str}_h{hidden_feature_list}_lr{learning_rate}"
    #             print(model_name)
    #             multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)


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

                    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000,
                                                  sklearn_random=sklearn_random, criterion_type="MSE",
                                                  hidden_feature=hidden_feature, learning_rate=learning_rate,
                                                  optimizer_str=optimizer_str, model_str="MMoE", config=config)
                    model_name = f"MMoE_MSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}_nexp{num_experts}_uexp{expert_unit}_utower{hidden_units}"

                    print(model_name)
                    multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    # hidden_feature = 4
    # optimizer_str = "Adam"
    # learning_rate = 0.001
    # y_pred = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000,
    #                               sklearn_random=sklearn_random, criterion_type="MSE",
    #                               hidden_feature=hidden_feature, learning_rate=learning_rate,
    #                               optimizer_str=optimizer_str, model_str="FCN")
    # model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
    # print(model_name)
    # multi_eval(y_test, y_pred, result_full_filepath, model_name)

    """
    hidden_feature = 4
    optimizer_str = "Adam"
    learning_rate = 0.001
    y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 2000,
                                  sklearn_random=sklearn_random, criterion_type="MSE",
                                  hidden_feature=hidden_feature, learning_rate=learning_rate,
                                  optimizer_str=optimizer_str, model_str="MoE")
    model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
    print(model_name)
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
                y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 5000, sklearn_random=sklearn_random, criterion_type="MSE", hidden_feature=hidden_feature, learning_rate=learning_rate, optimizer_str=optimizer_str)
                model_name = f"NeuralNetworkMSE_opt{optimizer_str}_h{hidden_feature}_lr{learning_rate}"
                print(model_name)
                multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)
    # optimizer_str = "Adagrad"
    # for max_epoch in [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 35, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000]:
    #     # Neural Network
    #     hidden_feature = 96
    #     learning_rate = 0.001
    #     y_pred, end_epoch = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, max_epoch, sklearn_random=sklearn_random, criterion_type="MSE", hidden_feature=hidden_feature, learning_rate=learning_rate,
    #                                   earlystop_turn_on=False, val_ratio=0, optimizer_str=optimizer_str)
    #     model_name = f"Neural Network MSE epoch{max_epoch}_opt{optimizer_str}"
    #     print(model_name)
    #     multi_eval(y_test, y_pred, result_full_filepath, model_name, end_epoch)

    # lgb multi-class
    gbm = lgb.LGBMRegressor(objective='multiclass', num_leaves=31, learning_rate=0.05, num_classes=num_classes)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    model_name = "lightgbm"
    print(model_name)
    multi_eval(y_test, y_pred, result_full_filepath, model_name)

    f = open(result_full_filepath, "a+")
    f.write("\n")
    f.close()
