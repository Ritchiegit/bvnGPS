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
def multi_eval(y_test, y_pred, result_save_path, model_name=None):
    y_pred_index = np.argmax(y_pred, axis=1)
    AUC_0 = metrics.roc_auc_score(1*(y_test == 0), 1*(y_pred_index == 0))
    AUC_1 = metrics.roc_auc_score(1*(y_test == 1), 1*(y_pred_index == 1))
    AUC_2 = metrics.roc_auc_score(1*(y_test == 2), 1*(y_pred_index == 2))
    print(f"{model_name},Health{AUC_0},Bacteria{AUC_1},Virus{AUC_2}\n")
    f = open(result_save_path, "a+")
    f.write(f"{model_name},{AUC_0},{AUC_1},{AUC_2}\n")

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
    multi_eval(y_test, y_pred_onehot, result_full_filepath, model_name)

    # lgb multi-class
    gbm = lgb.LGBMRegressor(objective='multiclass', num_leaves=31, learning_rate=0.05, num_classes=num_classes)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    model_name = "lightgbm"
    multi_eval(y_test, y_pred, result_full_filepath, model_name)

    # Neural Network
    y_pred = NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, 10000, sklearn_random=sklearn_random, criterion_type="MSE")
    model_name = "Neural Network MSE"
    multi_eval(y_test, y_pred, result_full_filepath, model_name)

    f = open(result_full_filepath, "a+")
    f.write("\n")
    f.close()