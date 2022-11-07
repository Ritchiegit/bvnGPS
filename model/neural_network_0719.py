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
from model.three_input_model import three_input_model_add, three_input_model_concatenate
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
    """
    仅对三分类loss进行训练
    """
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

def pred_subnet(model, X_test):
    X_test_tc = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tc)[0].cpu().detach().numpy()
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
                         criterion_type="MSE", batch_size=128, learning_rate=0.001,
                         earlystop_turn_on=True, val_ratio=0.2, optimizer_str="Adam", model_str="FCN", config=None,
                         hidden_feature_list=None, model_save_folder_path="weights"):

    optimizer_setting = f"NN_opt{optimizer_str}_lr{learning_rate}"

    if val_ratio != 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio,
                                                          random_state=sklearn_random)
    else:
        X_val = np.array([0])
        y_val = np.array([0])
    print("config 101", config)
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
            exit(1)
            return
        model = mutli_layers_FCN(input_feature=X_train.shape[1], hidden_feature_list=hidden_feature_list, output_feature=num_classes)
    elif model_str == "three_input_model_add":
        model = three_input_model_add(b_num = config['bacteria'], v_num = config['virus'], n_num = config['noninfected'], hidden_feature=config['hidden_feature'], classifier_feature_list=config['classifier_feature_list'], output_feature=3)
        print("three_input_model_add")
    elif model_str == "three_input_model_concatenate":
        model = three_input_model_concatenate(b_num = config['bacteria'], v_num = config['virus'], n_num = config['noninfected'], hidden_feature=config['hidden_feature'], classifier_feature_list=config['classifier_feature_list'], output_feature=3)
        print("three_input_model_concatenate")
    else:
        print("model is not FCN or MoE")
        exit(1)
        return
    model_name_for_save = optimizer_setting + "_" + model.get_algorithm_file_name()

    model = model.to(device)
    print(model)
    if optimizer_str == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    elif optimizer_str == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        print("please check optimizer")
        return

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    if criterion_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("please check criterion")
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
        print("model_save_folder_path", model_save_folder_path)
        print("model_name_for_save", model_name_for_save)

        torch.save(model, f"{model_save_folder_path}/{model_name_for_save}_val{val_loss}_epoch{end_epoch}.pth")
    y_pred = pred(model=model, X_test=X_test)
    return y_pred, end_epoch, model_name_for_save


def evaluate_subnet(loader, model, criterion, num_classes=2):
    """
    对所有二分类loss和三分类loss进行评估
    """
    model.eval()
    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(loader):
        x, y = batch
        bs = y.shape[0]
        total += bs
        y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        loss_three = criterion(output[0], y)
        loss_bacteria = criterion(output[1], y_bacteria)
        loss_virus = criterion(output[2], y_virus)
        loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_three + loss_bacteria + loss_virus + loss_noninfected
        # y = torch.eye(num_classes)[y].to(device)
        # output = model(x)
        # loss_batch = criterion(output[0], y)
        loss += bs * loss_batch
    average_loss = loss.cpu().detach().numpy() / total
    return average_loss

def train_subnet(train_loader, model, criterion, optimizer, val_loader=None, num_classes=2, eval_mode='only_three'):
    """
    对所有二分类loss和三分类loss进行训练
    """
    model.train()  # 启用dropout and batch normalization
    loss = 0.
    total = 0.
    val_loss = None
    for i_batch, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        bs = y.shape[0]
        total += bs
        y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        loss_three = criterion(output[0], y)
        loss_bacteria = criterion(output[1], y_bacteria)
        loss_virus = criterion(output[2], y_virus)
        loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_three + loss_bacteria + loss_virus + loss_noninfected
        loss_batch.backward()
        optimizer.step()
        loss += bs * loss_batch
        # print(f"batch:{i_batch}, loss:{loss_batch}")

    # TODO 0916
    # print(eval_mode)
    if eval_mode == "only_three":

        train_loss = evaluate_subnet_three(train_loader, model, criterion, num_classes=num_classes)
        if val_loader is not None:
            # TODO 0916
            val_loss = evaluate_subnet_three(val_loader, model, criterion, num_classes=num_classes)
        # print("in three")
    elif eval_mode == "three_and_binary":
        train_loss = evaluate_subnet(train_loader, model, criterion, num_classes=num_classes)
        if val_loader is not None:
            # TODO 0916
            val_loss = evaluate_subnet(val_loader, model, criterion, num_classes=num_classes)
        # print("in all")
    else:
        print("eval_mode", eval_mode)
        exit(1)

    return train_loss, val_loss

# TODO
def NN_2layer_subnet_train_test(X_train_ori, X_test, y_train_ori, y_test, num_classes, max_epochs=10000, sklearn_random=109,
                         criterion_type="MSE", batch_size=128, learning_rate=0.001,
                         earlystop_turn_on=True, val_ratio=0.2, optimizer_str="Adam", model_str="FCN", config=None,
                         hidden_feature_list=None, model_save_folder_path="weights"):
    optimizer_setting = f"NN_opt{optimizer_str}_lr{learning_rate}"

    if val_ratio != 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train_ori, y_train_ori, test_size=val_ratio,
                                                          random_state=sklearn_random)
    else:
        X_val = np.array([0])
        y_val = np.array([0])
    """
    print("X_train_ori.shape", X_train_ori.shape)
    for train_data_each in X_train_ori_three_categories_list:
        print("train_data_each", train_data_each.shape)
    # X_train_ori_three_categories_list

    # np.save("0830_train.npy",[X_train_ori_three_categories_list, X_train_ori, y_train_ori])

    # test 用不上，在外面进行评估，处理train val即可。 X_train_ori_three_categories_list, X_train_ori, y_train_ori 都要进行一个划分
    if val_ratio != 0:  # train: val(早停) = 4:1; train+val: test = 7:3  # X_train_ori_three_categories_list = 7
        split_result = train_test_split(*X_train_ori_three_categories_list, X_train_ori, y_train_ori, test_size=val_ratio,
                                                          random_state=sklearn_random)
    else:
        X_val = np.array([0])
        y_val = np.array([0])

    assert len(split_result) == (len(X_train_ori_three_categories_list) + 2) * 2
    X_train = split_result[-4]
    X_val = split_result[-3]
    y_train = split_result[-2]
    y_val = split_result[-1]
    X_train_three_categories_list = []
    X_val_three_categories_list = []
    assert X_train.shape[0] == int(X_train_ori.shape[0] * (1 - val_ratio)), print(X_train.shape[0], int(X_train_ori.shape[0] * (1 - val_ratio)))
    for i in range(len(X_train_ori_three_categories_list)):
        X_train_three_categories_list.append(split_result[i * 2])
        X_val_three_categories_list.append(split_result[i * 2 + 1])

    print("X_train, X_val", X_train.shape, X_val.shape)  # 1418*56 355*56
    print("y_train, y_val", y_train.shape, y_val.shape)  # 1418 355 760
    print("len", len(X_train_three_categories_list))
    print("len", len(X_val_three_categories_list))
    for X_train_three_categories_each in X_train_three_categories_list:
        print("X_train", X_train_three_categories_each.shape)  # 1418 * 21 24 11
    for X_val_three_categories_each in X_val_three_categories_list:
        print("X_val", X_val_three_categories_each.shape)  # 355 * 21 24 11

    tmp_X_train = np.concatenate(X_train_three_categories_list, axis=1)
    tmp_X_val = np.concatenate(X_val_three_categories_list, axis=1)
    assert np.array_equal(tmp_X_train, X_train)
    assert np.array_equal(tmp_X_val, X_val)
    # X_train, X_val, X_test  # (1418, 56) (355, 56) (760, 56)
    # X_train_three_categories  # (1418, 21) (1418, 24) (1418, 11)
    # X_val_three_categories  # (355, 21) (355, 24) (355, 11)
    # y_train, y_val, y_test  # 1418 355 760
    
    # X_train_tc 1418, 56
    # X_val_tc 355, 56
    X_train_three_categories_tc_each torch.Size([1418, 21])
    X_train_three_categories_tc_each torch.Size([1418, 24])
    X_train_three_categories_tc_each torch.Size([1418, 11])
    X_val_three_categories_tc_each torch.Size([355, 21])
    X_val_three_categories_tc_each torch.Size([355, 24])
    X_val_three_categories_tc_each torch.Size([355, 11])
    y_train_tc 1418 
    y_val_tc 355

    X_train_three_categories_tc_list = []
    X_val_three_categories_tc_list = []
    for X_train_three_categories_each in X_train_three_categories_list:
        X_train_three_categories_tc_list.append(torch.from_numpy(X_train_three_categories_each).to(device))
    for X_val_three_categories_each in X_val_three_categories_list:
        X_val_three_categories_tc_list.append(torch.from_numpy(X_val_three_categories_each).to(device))

    for X_train_three_categories_tc_each in X_train_three_categories_tc_list:
        print("X_train_three_categories_tc_each", X_train_three_categories_tc_each.shape)
    for X_val_three_categories_tc_each in X_val_three_categories_tc_list:
        print("X_val_three_categories_tc_each", X_val_three_categories_tc_each.shape)
    """

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
            exit(1)
            return
        model = mutli_layers_FCN(input_feature=X_train.shape[1], hidden_feature_list=hidden_feature_list, output_feature=num_classes)
    elif model_str == "three_input_model_add":
        model = three_input_model_add(b_num = config['bacteria'], v_num = config['virus'], n_num = config['noninfected'], hidden_feature=config['hidden_feature'], classifier_feature_list=config['classifier_feature_list'], output_feature=3)
        print("three_input_model_add")
    elif model_str == "three_input_model_concatenate":
        model = three_input_model_concatenate(b_num = config['bacteria'], v_num = config['virus'], n_num = config['noninfected'], hidden_feature=config['hidden_feature'], classifier_feature_list=config['classifier_feature_list'], output_feature=3)
        print("three_input_model_concatenate")
    elif model_str == "three_input_model_concatenate_subnet":
        from model.three_input_model_subnet import three_input_model_concatenate_subnet_train
        model = three_input_model_concatenate_subnet_train(b_num = config['bacteria'], v_num = config['virus'], n_num = config['noninfected'], hidden_feature=config['hidden_feature'], classifier_feature_list=config['classifier_feature_list'], output_feature=3)
        # 同时训练4个loss
    else:
        print(model_str)
        print("three_input_model_concatenate_subnet_seperate_train")
        print("model is not FCN or MoE")
        exit(1)
        return
    model_name_for_save = optimizer_setting + "_" + model.get_algorithm_file_name()

    model = model.to(device)
    print(model)
    if optimizer_str == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    elif optimizer_str == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        print("please check optimizer")
        return

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    if criterion_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("please check criterion")
        return
    es = earlystop()
    min_val_loss = 1e10
    end_epoch = max_epochs
    with tqdm(range(max_epochs)) as t:
        for i_epoch in t:  # tqdm(range(max_epochs)):
            if val_ratio == 0:
                train_loss, val_loss = train_subnet(train_loader, model, criterion, optimizer=optimizer, val_loader=None,
                                             num_classes=num_classes)
                t.set_description(f"Epoch:{i_epoch}, train loss: {train_loss:.10f}")
            else:
                train_loss, val_loss = train_subnet(train_loader, model, criterion, optimizer=optimizer, val_loader=val_loader,
                                             num_classes=num_classes, eval_mode=config['eval_mode'])
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
        print("model_save_folder_path", model_save_folder_path)
        print("model_name_for_save", model_name_for_save)

        torch.save(model, f"{model_save_folder_path}/{model_name_for_save}_val{val_loss}_epoch{end_epoch}.pth")
    y_pred = pred_subnet(model=model, X_test=X_test)
    return y_pred, end_epoch, model_name_for_save

def evaluate_subnet_binary(loader, model, criterion, num_classes=2):
    """
    仅对所有二分类loss进行评估
    """
    model.eval()
    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(loader):
        x, y = batch
        bs = y.shape[0]
        total += bs
        y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        # y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        # loss_three = criterion(output[0], y)
        loss_bacteria = criterion(output[1], y_bacteria)
        loss_virus = criterion(output[2], y_virus)
        loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_bacteria + loss_virus + loss_noninfected
        # y = torch.eye(num_classes)[y].to(device)
        # output = model(x)
        # loss_batch = criterion(output[0], y)
        loss += bs * loss_batch
    average_loss = loss.cpu().detach().numpy() / total
    return average_loss

def train_subnet_binary(train_loader, model, criterion, optimizer, val_loader=None, num_classes=2):
    """
    仅对所有二分类loss进行训练
    """
    model.train()  # 启用dropout and batch normalization
    loss = 0.
    total = 0.
    val_loss = None
    for i_batch, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        bs = y.shape[0]
        total += bs
        y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        # y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        # loss_three = criterion(output[0], y)
        loss_bacteria = criterion(output[1], y_bacteria)
        loss_virus = criterion(output[2], y_virus)
        loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_bacteria + loss_virus + loss_noninfected
        loss_batch.backward()
        optimizer.step()
        loss += bs * loss_batch
        # print(f"batch:{i_batch}, loss:{loss_batch}")

    train_loss = evaluate_subnet_binary(train_loader, model, criterion, num_classes=num_classes)
    if val_loader is not None:
        val_loss = evaluate_subnet_binary(val_loader, model, criterion, num_classes=num_classes)
    return train_loss, val_loss

def evaluate_subnet_three(loader, model, criterion, num_classes=2):
    """
    仅对最终三分类loss进行评估
    """
    model.eval()
    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(loader):
        x, y = batch
        bs = y.shape[0]
        total += bs
        # y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        # y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        # y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        loss_three = criterion(output[0], y)
        # loss_bacteria = criterion(output[1], y_bacteria)
        # loss_virus = criterion(output[2], y_virus)
        # loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_three
        # y = torch.eye(num_classes)[y].to(device)
        # output = model(x)
        # loss_batch = criterion(output[0], y)
        loss += bs * loss_batch
    average_loss = loss.cpu().detach().numpy() / total
    return average_loss

def train_subnet_three(train_loader, model, criterion, optimizer, val_loader=None, num_classes=2):
    """
    仅用最后三分类作为loss，进行训练
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param val_loader:
    :param num_classes:
    :return:
    """
    model.train()  # 启用dropout and batch normalization
    loss = 0.
    total = 0.
    val_loss = None
    for i_batch, batch in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = batch
        bs = y.shape[0]
        total += bs
        # y_bacteria = torch.eye(2)[1 * (y == 1)].to(device)
        # y_virus = torch.eye(2)[1 * (y == 2)].to(device)
        # y_noninfected = torch.eye(2)[1 * (y == 0)].to(device)
        y = torch.eye(num_classes)[y].to(device)
        output = model(x)
        loss_three = criterion(output[0], y)
        # loss_bacteria = criterion(output[1], y_bacteria)
        # loss_virus = criterion(output[2], y_virus)
        # loss_noninfected = criterion(output[3], y_noninfected)
        loss_batch = loss_three
        loss_batch.backward()
        optimizer.step()
        loss += bs * loss_batch
        # print(f"batch:{i_batch}, loss:{loss_batch}")

    train_loss = evaluate_subnet_three(train_loader, model, criterion, num_classes=num_classes)
    if val_loader is not None:
        val_loss = evaluate_subnet_three(val_loader, model, criterion, num_classes=num_classes)
    return train_loss, val_loss

def NN_2layer_subnet_seperate_train_test(X_train_ori, X_test, y_train_ori, y_test, num_classes, max_epochs=10000,
                                sklearn_random=109,
                                criterion_type="MSE", batch_size=128, learning_rate=0.001,
                                earlystop_turn_on=True, val_ratio=0.2, optimizer_str="Adam", model_str="FCN",
                                config=None,
                                hidden_feature_list=None, model_save_folder_path="weights"):
    optimizer_setting = f"NN_opt{optimizer_str}_lr{learning_rate}"

    if val_ratio != 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train_ori, y_train_ori, test_size=val_ratio,
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

    if model_str == "three_input_model_concatenate_subnet_seperate_train":
        from model.three_input_model_subnet import three_input_model_concatenate_subnet_train
        model = three_input_model_concatenate_subnet_train(b_num=config['bacteria'], v_num=config['virus'],
                                                           n_num=config['noninfected'],
                                                           hidden_feature=config['hidden_feature'],
                                                           classifier_feature_list=config['classifier_feature_list'],
                                                           output_feature=3)

    else:
        print("model_str is wrong:", model_str)
        exit(1)
        return


    model_name_for_save = optimizer_setting + "_" + model.get_algorithm_file_name()

    model = model.to(device)
    print(model)
    if optimizer_str == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    elif optimizer_str == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        print("please check optimizer")
        return

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)  # 之前都是0.0001
    if criterion_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == "MSE":
        criterion = nn.MSELoss()
    else:
        print("please check criterion")
        return

    es = earlystop()
    min_val_loss = 1e10
    end_epoch = max_epochs
    with tqdm(range(max_epochs)) as t:
        for i_epoch in t:  # tqdm(range(max_epochs)):
            if val_ratio == 0:
                train_loss, val_loss = train_subnet_binary(train_loader, model, criterion, optimizer=optimizer,
                                                    val_loader=val_loader,
                                                    num_classes=num_classes)
                t.set_description(f"Epoch:{i_epoch}, train loss: {train_loss:.10f}")
            else:
                # TODO
                train_loss, val_loss = train_subnet_binary(train_loader, model, criterion, optimizer=optimizer,
                                                    val_loader=val_loader,
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

    # freeze feature parameters
    cnt = 0
    for child in model.children():
        cnt += 1
        print(cnt)
        for param in child.parameters():
            param.requires_grad = False
        print("child done", child)
        if cnt >= 6: break

    # cnt = 0
    # for child in model.children():
    #     cnt += 1
    #     print("cnt", cnt)
    #     for param in child.parameters():
    #         print(param.requires_grad)

    # if val_loader is not None:
    #     val_loss_binary = evaluate_subnet_binary(val_loader, model, criterion, num_classes=num_classes)
    # print("val_loss_binary", val_loss_binary)

    es = earlystop()
    min_val_loss = 1e10
    end_epoch = max_epochs
    with tqdm(range(max_epochs)) as t:
        for i_epoch in t:  # tqdm(range(max_epochs)):
            if val_ratio == 0:
                train_loss, val_loss = train_subnet_three(train_loader, model, criterion, optimizer=optimizer,
                                                    val_loader=val_loader,
                                                    num_classes=num_classes)
                t.set_description(f"Epoch:{i_epoch}, train loss: {train_loss:.10f}")
            else:
                # TODO
                train_loss, val_loss = train_subnet_three(train_loader, model, criterion, optimizer=optimizer,
                                                    val_loader=val_loader,
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
        print("model_save_folder_path", model_save_folder_path)
        print("model_name_for_save", model_name_for_save)

        torch.save(model, f"{model_save_folder_path}/{model_name_for_save}_val{val_loss}_epoch{end_epoch}.pth")
    y_pred = pred_subnet(model=model, X_test=X_test)

    if val_loader is not None:
        val_loss_binary = evaluate_subnet_binary(val_loader, model, criterion, num_classes=num_classes)
    print("val_loss_binary", val_loss_binary)
    return y_pred, end_epoch, model_name_for_save
