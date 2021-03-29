import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
from load_data.load_data_raw import load_data_raw
from data_processing.process_data_label import get_label_multilabel
from sklearn.model_selection import train_test_split

pred_path = "../label_for_validation/pred_only_21802_57065.csv"
label_path = "../label_for_validation/label_only_21802_57065.csv"

#
# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.3.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.3.csv"

# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.7.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.7.csv"



pred_3 = pd.read_csv(pred_path, header=None)
pred_3 = pred_3.values
label = pd.read_csv(label_path, header=None)
label = label.values.flatten()

print(pred_3)
print(label)
plt.figure(figsize=(4, 4), dpi=300)

for i in range(2, 3):
    label_one = (label == i) * 1
    # pred_one = (pred == i) * 1
    pred_one = pred_3[:, i]
    # pred_one = (pred_3.argmax(axis=1) == i)

    # print(label_one.shape)
    # print(pred_one.shape)
    pred_one = pred_one.flatten()

    fpr, tpr, _ = roc_curve(label_one, pred_one)
    roc_auc = auc(fpr, tpr)
    lw = 1
    plt.plot(fpr, tpr, lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{pred_path[:-4]}.png')
plt.show()

plt.close()