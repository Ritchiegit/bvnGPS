import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # TODO cpu

class FCN(nn.Module):
    def __init__(self, input_feature=22583, hidden_feature=256, output_feature=2):
        super(FCN, self).__init__()
        self.hidden_feature = hidden_feature
        self.output_feature = output_feature
        self.linear = nn.Linear(input_feature, hidden_feature, bias=True).to(device)
        self.bn = nn.BatchNorm1d(self.hidden_feature).to(device)
        self.linear2 = nn.Linear(hidden_feature, output_feature, bias=True).to(device)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x