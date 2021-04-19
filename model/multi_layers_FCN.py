from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class mutli_layers_FCN(nn.Module):
    def __init__(self, input_feature=22583, hidden_feature_list=[4, 4], output_feature=2):
        super(mutli_layers_FCN, self).__init__()
        self.output_feature = output_feature
        self.hidden_layer_list = []
        self.bn_layer_list = []
        units_in_last_layer = input_feature
        for hidden_feature in hidden_feature_list:
            print("last, now", units_in_last_layer, hidden_feature)
            # units_in_last_layer # hidden_feature
            linear = nn.Linear(units_in_last_layer, hidden_feature, bias=True).to(device)
            bn = nn.BatchNorm1d(hidden_feature).to(device)
            self.hidden_layer_list.append(linear)
            self.bn_layer_list.append(bn)
            units_in_last_layer = hidden_feature
        self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list)
        self.bn_layer_list = nn.ModuleList(self.bn_layer_list)
        # units_in_last_layer output_feature
        self.last_layer_linear = nn.Linear(units_in_last_layer, output_feature, bias=True).to(device)
        # softmax

    def forward(self, x):

        for hidden_layer, bn in zip(self.hidden_layer_list, self.bn_layer_list):
            x = hidden_layer(x)
            x = F.relu(x)
            x = bn(x)

        x = self.last_layer_linear(x)
        # x = F.softmax(x, dim=1)
        x = F.sigmoid(x)
        return x


"""
1. 
input Linear hidden1
bn
hidden1 Linear hidden2
bn
---
hidden2 Linear output
softmax

2. 
input Linear hidden1
bn
---
hidden1 Linear output
softmax
"""
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    dataset = pd.read_csv('../learning/iris_data.csv')
    dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2
    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                        dataset.species.values, test_size=0.8)
    import numpy as np

    # wrap up with Variable in pytorch
    train_X = torch.from_numpy(train_X.astype(np.float32)).to(device)
    test_X = torch.from_numpy(test_X.astype(np.float32)).to(device)
    train_y = torch.from_numpy(train_y.astype(np.int64)).to(device)
    test_y = torch.from_numpy(test_y.astype(np.int64)).to(device)
    net = mutli_layers_FCN(input_feature=4, output_feature=3)
    criterion = nn.CrossEntropyLoss()  # cross entropy loss

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(10):
        optimizer.zero_grad()
        out = net(train_X)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        input()
        if epoch % 100 == 0:
            print('number of epoch', epoch, 'loss', loss)

    predict_out = net(test_X)
    _, predict_y = torch.max(predict_out, 1)
    test_y_cpu = test_y.cpu()
    predict_y_cpu = predict_y.cpu()

    print('prediction accuracy', accuracy_score(test_y_cpu, predict_y_cpu))
    print('macro precision', precision_score(test_y_cpu, predict_y_cpu, average='macro'))
    print('micro precision', precision_score(test_y_cpu, predict_y_cpu, average='micro'))
    print('macro recall', recall_score(test_y_cpu, predict_y_cpu, average='macro'))
    print('micro recall', recall_score(test_y_cpu, predict_y_cpu, average='micro'))
