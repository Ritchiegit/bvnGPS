# coding: UTF-8
import torch
import torch.nn as nn

class Config(object):
    def __init__(self, num_experts=8, expert_unit=16, hidden_units=8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_dict = [1,1,1]  # ok
        self.num_feature = 49  # input feature ok
        self.num_tasks = 3  # ok
        self.num_experts = num_experts
        self.units = expert_unit  # expert units
        self.hidden_units = hidden_units  # tower units

        # self.model_name = 'mmoe'
        # self.train_path = data_dir + 'train.txt'
        # self.test_path = data_dir + 'test.txt'
        # self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        # self.require_improvement = 1000
        # self.dropout = 0.5
        # self.learning_rate = 1e-4
        # self.label_columns = ['income_50k', 'marital_stat']
        # self.embed_size = 64
        # self.batch_size = 128
        # self.field_size = 0
        # self.num_epochs = 100




class MMoE(nn.Module):
    def __init__(self,config):
        super(MMoE, self).__init__()

        # accept_unit = config.field_size*config.embed_size
        accept_unit = config.num_feature
        w = torch.empty(accept_unit, config.units, config.num_experts,device=config.device)

        self.expert_kernels = torch.nn.Parameter(torch.nn.init.xavier_normal_(w), requires_grad=True)
        w = torch.empty(accept_unit, config.num_experts, device=config.device)
        self.gate_kernels = torch.nn.ParameterList(
            [nn.Parameter(torch.nn.init.xavier_normal_(w), requires_grad=True) for i in
             range(config.num_tasks)])

        self.expert_kernels_bias = torch.nn.Parameter(torch.zeros(config.units, config.num_experts, device=config.device),
                                                      requires_grad=True)
        self.gate_kernels_bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(config.num_experts, device=config.device), requires_grad=True) for i in range(config.num_tasks)])

        self.output_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(config.units,config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units,unit),
        )
            for unit in config.label_dict
        ])
        # print("*************************************self.output_layer.lens", len(self.output_layer))
        self.expert_activation = nn.ReLU()
        self.gate_activation = nn.Softmax(dim=-1)

        # self.embedding_layer = nn.Embedding(config.num_feature,config.embed_size)

    def forward(self,x):
        gate_outputs = []
        final_outputs = []
        # xi =x[0]
        # xv = x[1]

        # self.embeddings = self.embedding_layer(xi)
        # feat_value = xv.view(-1,xv.size(1),1)
        #
        # self.embeddings = feat_value * self.embeddings
        # self.embeddings = self.embeddings.view(xv.size(0),-1)

        # print("x_shape", x.shape)
        # print("self.expert_kernels.shape", self.expert_kernels.shape)
        expert_outputs = torch.einsum("ab,bcd->acd", (x, self.expert_kernels))
        expert_outputs += self.expert_kernels_bias
        expert_outputs = self.expert_activation(expert_outputs)

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = torch.einsum("ab,bc->ac", (x, gate_kernel))
            gate_output += self.gate_kernels_bias[index]
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = expert_outputs * expanded_gate_output.expand_as(expert_outputs)
            final_outputs.append(torch.sum(weighted_expert_output, 2))
        # print("len(gate_output)", len(gate_output))
        output_layers = []
        for i, output in enumerate(final_outputs):
            output_layers.append(self.output_layer[i](output))

        # print("output_layers[0].shape", output_layers[0].shape)
        output = torch.cat(output_layers, dim=1)
        # print("output.shape", output.shape)

        return output



