from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from model.FCN import FCN
from model.MoE import MoE
from model.mmoe import MMoE
from model.mmoe import Config
from model.multi_layers_FCN import mutli_layers_FCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # TODO cpu






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
        # print(f"batch:{i_batch}, loss:{loss_batch}")

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
            for i in range(len(self.last_loss) - 1):
                if self.last_loss[i] > self.last_loss[i + 1]:  # 有降低说明还没有一直升高。
                    always_go_higher = False
                    break
            # if always_go_higher == True:
            # print(self.last_loss)
        return always_go_higher


def NN_2layer_train_test(X_train, X_test, y_train, y_test, num_classes, max_epochs=10000, sklearn_random=109,
                         criterion_type="MSE", hidden_feature=256, batch_size=128, learning_rate=0.001,
                         earlystop_turn_on=True, val_ratio=0.2, optimizer_str="Adam", model_str="FCN", config=None,
                         hidden_feature_list=None, model_save_folder_path="weights",  model_name_for_save=None):
    if val_ratio != 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio,
                                                          random_state=sklearn_random)
    else:
        X_val = np.array([0])
        y_val = np.array([0])
    X_train_tc, X_val_tc, y_train_tc, y_val_tc = torch.from_numpy(X_train).to(device), torch.from_numpy(X_val).to(
        device), torch.from_numpy(y_train).to(device), torch.from_numpy(y_val).to(device)
    data_train_tc_tensorDataset = TensorDataset(X_train_tc, y_train_tc)
    data_val_tc_tensorDataset = TensorDataset(X_val_tc, y_val_tc)
    train_loader = DataLoader(data_train_tc_tensorDataset, batch_size=batch_size)
    val_loader = DataLoader(data_val_tc_tensorDataset, batch_size=batch_size)
    # print("NN input feature number", X_train.shape[1])
    # print("NN output feature", num_classes)

    if model_str == "FCN":
        print("in FCN")
        model = FCN(input_feature=X_train.shape[1], hidden_feature=hidden_feature, output_feature=num_classes)
    elif model_str == "MoE":
        print("in MoE")
        model = MoE(input_feature=X_train.shape[1], hidden_feature=hidden_feature, output_feature=num_classes)
    elif model_str == "MMoE":
        print("in MMoE")
        if config is None:
            config = Config()
        config.num_feature = X_train.shape[1]
        model = MMoE(config=config)  # config=Config()
    elif model_str == "multi_layers_FCN":
        print("in multi_layers_FCN")
        if hidden_feature_list == None:
            print("please")
            input()
            exit(1)
            return
        model = mutli_layers_FCN(input_feature=X_train.shape[1], hidden_feature_list=hidden_feature_list, output_feature=num_classes)
    else:
        print("model is not FCN or MoE")
        input()
        exit(1)
        return

    model = model.to(device)
    if optimizer_str == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    elif optimizer_str == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        print("please check optimizer")
        input()
        return

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    if criterion_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("please check criterion")
        input()
        return
    es = earlystop()
    min_val_loss = 1e10
    end_epoch = max_epochs
    with tqdm(range(max_epochs)) as t:

        for i_epoch in t:  # tqdm(range(max_epochs)):
            if val_ratio == 0:
                train_loss, val_loss = train(train_loader, model, criterion, optimizer=optimizer, val_loader=None,
                                             num_classes=num_classes)
                t.set_description(f"Epoch:{i_epoch}, train loss: {train_loss:.10f}")
            else:
                train_loss, val_loss = train(train_loader, model, criterion, optimizer=optimizer, val_loader=val_loader,
                                             num_classes=num_classes)
                t.set_description(f"Epoch:{i_epoch}, train loss: {train_loss:.10f}, val loss: {val_loss:.10f}")
                if val_loss < min_val_loss:
                    # torch.save(model.state_dict(), f"weights/FCN_i{X_train.shape[1]}_h{hidden_feature}_bs{batch_size}_lr{learning_rate}_val{val_loss}")
                    min_val_loss = val_loss
                if earlystop_turn_on == False:
                    continue
                always_go_higher = es.save_loss_and_check_it_is_always_go_higher(val_loss)
                if always_go_higher == True:
                    end_epoch = i_epoch
                    break
        # torch.save(model.state_dict(), f"{model_save_folder_path}/{model_str}_i{X_train.shape[1]}_h{hidden_feature}_bs{batch_size}_lr{learning_rate}_val{val_loss}_epoch{end_epoch}.pth")
        torch.save(model, f"{model_save_folder_path}/{model_name_for_save}_val{val_loss}_epoch{end_epoch}.pth")
    y_pred = pred(model=model, X_test=X_test)
    return y_pred, end_epoch

# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
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
