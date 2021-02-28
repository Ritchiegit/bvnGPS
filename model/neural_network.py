from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    def __init__(self, input_feature=22583, hidden_feature=256, output_feature=2):
        super(FCN, self).__init__()
        self.hidden_feature = hidden_feature
        self.linear = nn.Linear(input_feature, hidden_feature, bias=True).to(device)
        self.bn = nn.BatchNorm1d(self.hidden_feature).to(device)
        self.linear2 = nn.Linear(hidden_feature, output_feature, bias=True).to(device)
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.linear2(x)
        x = F.softmax(x)
        return x

def evaluate(loader, model, criterion, num_classes=2):
    model.eval()
    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(loader):
        x, y = batch
        bs = y.shape[0]
        total += bs
        y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        loss_batch = criterion(output, y)
        loss += bs * loss_batch
    average_loss = loss.cpu().detach().numpy() / total
    return average_loss

def train(train_loader, model, criterion, optimizer, val_loader=None, num_classes=2):
    model.train()  # 启用dropout and batch normalization
    loss = 0.
    total = 0.
    val_loss = None
    for i_batch, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        bs = y.shape[0]
        total += bs

        y = torch.eye(num_classes)[y].to(device)
        output = model(x)

        loss_batch = criterion(output, y)
        loss_batch.backward()
        loss += bs * loss_batch
        optimizer.step()

    train_loss = evaluate(train_loader, model, criterion, num_classes=num_classes)
    if val_loader is not None:
        val_loss = evaluate(val_loader, model, criterion, num_classes=num_classes)
    return train_loss, val_loss

def pred(model, X_test):
    X_test_tc = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tc).cpu().detach().numpy()
    return y_pred

class earlystop():
    def __init__(self, number_of_go_higher_threshold=5):
        self.last_loss = []
        self.number_of_go_higher_threshold = number_of_go_higher_threshold

    def save_loss_and_check_it_is_always_go_higher(self, current_loss):
        self.last_loss.append(current_loss)
        always_go_higher = False
        if len(self.last_loss) > self.number_of_go_higher_threshold:
            self.last_loss.pop(0)
            always_go_higher = True
            for i in range(len(self.last_loss)-1):
                if self.last_loss[i] > self.last_loss[i+1]:  # 有降低说明还没有一直升高。
                    always_go_higher = False
                    break
            if always_go_higher == True:
                print(self.last_loss)
        return always_go_higher

def NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, max_batchs=10000, sklearn_random=109, criterion_type="MSE"):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=sklearn_random)
    X_train_tc, X_val_tc, y_train_tc, y_val_tc = torch.from_numpy(X_train).to(device), torch.from_numpy(X_val).to(device), torch.from_numpy(y_train).to(device), torch.from_numpy(y_val).to(device)
    data_train_tc_tensorDataset = TensorDataset(X_train_tc, y_train_tc)
    data_val_tc_tensorDataset = TensorDataset(X_val_tc, y_val_tc)
    train_loader = DataLoader(data_train_tc_tensorDataset, batch_size=128)
    val_loader = DataLoader(data_val_tc_tensorDataset, batch_size=128)
    print("NN input feature number", X_train.shape[1])
    print("NN output feature", num_classes)
    model = FCN(input_feature=X_train.shape[1], output_feature=num_classes)
    model = model.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.0001)
    if criterion_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("please check criterion")
        input()
        return
    es = earlystop()
    for batch in range(max_batchs):
        train_loss, val_loss = train(train_loader, model, criterion, optimizer=optimizer, val_loader=val_loader, num_classes=num_classes)
        print(f"Batch:{batch}, train loss: {train_loss}, val loss: {val_loss}")

        always_go_higher = es.save_loss_and_check_it_is_always_go_higher(val_loss)
        if always_go_higher == True:
            break
    y_pred = pred(model=model, X_test=X_test)
    return y_pred

# if __name__ == "__main__":
#     sklearn_random = 109
#     gene_GSE, label_GSE_concated = load_data_raw()
#     label = get_sick_label(label_GSE_concated)
#     import pandas as pd
#     num_classes = len(pd.Categorical(label).categories)
#     print("num_classes", num_classes)
#     path_data_without_process = "data_without_process.pickle"
#     if os.path.exists(path_data_without_process):
#         data = pickle.load(open(path_data_without_process, "rb"))
#     else:
#         print("输入字符 进行数据处理")
#         input()
#         from data_processing.process_data_label import get_data_without_process
#         gene_GSE, label_GSE_concated = load_data_raw()
#         data = get_data_without_process(gene_GSE, label)  # get_data_without_process(gene_GSE_adjusted_concated, gene_GSE)
#         pickle.dump(data, open(path_data_without_process, "wb"))
#     X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=sklearn_random)
#     NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, sklearn_random=sklearn_random)
#     #  train_eval(data, label, result_save_path, "without_process")
