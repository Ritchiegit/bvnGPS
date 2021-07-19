import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

# pred_path = "../label_for_validation/pred_only_21802_57065.csv"
# label_path = "../label_for_validation/label_only_21802_57065.csv"


# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.3.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.3.csv"
#
# pred_path = "../label_for_validation/pred_all_exclude_21802_570650.7.csv"
# label_path = "../label_for_validation/label_all_exclude_21802_570650.7.csv"

pred_path = "../label_for_validation/for_68310/pred_net.csv"
label_path = "../label_for_validation/for_68310/label_GSE68310.txt"


pred_3 = pd.read_csv(pred_path, header=None)
pred_3 = pred_3.values
label = pd.read_csv(label_path, header=None)
label = label.values.flatten()

print(pred_3)
print(label)
plt.figure(figsize=(4, 4), dpi=300)
class_num = ["Noninfected", "Bacterial", "Viral"]
for i in [1, 2, 0]:
    label_one = (label == i) * 1
    # pred_one = (pred == i) * 1
    pred_one = pred_3[:, i]
    # pred_one = (pred_3.argmax(axis=1) == i)

    # print(label_one.shape)
    # print(pred_one.shape)
    pred_one = pred_one.flatten()

    fpr, tpr, _ = roc_curve(label_one, pred_one)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    lw = 1
    plt.plot(fpr, tpr, lw=lw, label=f'{class_num[i]} (AUC=%0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    plt.grid()
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{pred_path[:-4]}.png')
plt.show()

plt.close()