import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoE(nn.Module):
    def __init__(self, input_feature=22583, hidden_feature=256, output_feature=2, Expert_num=8):

        # def __init__(self, Expert_num=8):
        super(MoE, self).__init__()
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.output_feature = output_feature
        linears_temp = []
        linears_2_temp = []
        bns_temp = []
        for i in range(Expert_num):
            linears_temp.append(nn.Linear(input_feature, hidden_feature, bias=True))
            linears_2_temp.append(nn.Linear(hidden_feature, output_feature, bias=True))
            bns_temp.append(nn.BatchNorm1d(self.hidden_feature))

        self.linears = nn.ModuleList(linears_temp).to(device)
        self.linears_2 = nn.ModuleList(linears_2_temp).to(device)
        self.bns = nn.ModuleList(bns_temp).to(device)
    def forward(self, x_raw):
        x_out_all = 0
        for i in range(len(self.linears)):
            x = self.linears[i](x_raw)
            x = F.relu(x)
            # x = self.bn(x)
            x = self.bns[i](x)
            x_out = self.linears_2[i](x)
            x_out_all += x_out
        x_out_all = F.softmax(x_out_all)
        # x_out_all = F.sigmoid(x_out_all)
        return x_out_all