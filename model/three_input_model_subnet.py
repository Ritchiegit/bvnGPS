import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # TODO cpu

class three_input_model_concatenate_subnet_train(nn.Module):
    def __init__(self, b_num, v_num, n_num, hidden_feature, classifier_feature_list=[], output_feature=3):
        super(three_input_model_concatenate_subnet_train, self).__init__()

        self.bacteria_num = b_num
        self.virus_num = v_num
        self.noninfected_num = n_num
        self.hidden_feature = hidden_feature
        self.classifier_feature_list = classifier_feature_list
        self.output_feature = output_feature

        self.linear_bacteria = nn.Sequential(
            nn.Linear(self.bacteria_num, self.hidden_feature, bias=True),
            nn.BatchNorm1d(self.hidden_feature),
            nn.ReLU()
        ).to(device)
        self.linear_virus = nn.Sequential(
            nn.Linear(self.virus_num, self.hidden_feature, bias=True),
            nn.BatchNorm1d(self.hidden_feature),
            nn.ReLU()
        ).to(device)
        self.linear_noninfected = nn.Sequential(
            nn.Linear(self.noninfected_num, self.hidden_feature, bias=True),
            nn.BatchNorm1d(self.hidden_feature),
            nn.ReLU()
        ).to(device)

        self.binary_classifier_bacteria = nn.Sequential(
            nn.Linear(hidden_feature, 2, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.binary_classifier_virus = nn.Sequential(
            nn.Linear(hidden_feature, 2, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.binary_classifier_noninfected = nn.Sequential(
            nn.Linear(hidden_feature, 2, bias=True),
            nn.Sigmoid()
        ).to(device)


        self.output_classifier = nn.Sequential()
        if len(classifier_feature_list) == 0:
            self.output_classifier.add_module("output", nn.Linear(hidden_feature * 3, output_feature))
        else:
            for i in range(len(classifier_feature_list)):
                if i == 0:
                    self.output_classifier.add_module(f"lin_{i}", nn.Linear(hidden_feature * 3, classifier_feature_list[i]))
                    self.output_classifier.add_module(f"bn_{i}", nn.BatchNorm1d(classifier_feature_list[i]))
                    self.output_classifier.add_module(f"relu_{i}", nn.ReLU())

                elif i == len(classifier_feature_list) - 1:
                    self.output_classifier.add_module(f"lin_{i}", nn.Linear(classifier_feature_list[i - 1], classifier_feature_list[i]))
                    self.output_classifier.add_module(f"bn_{i}", nn.BatchNorm1d(classifier_feature_list[i]))
                    self.output_classifier.add_module(f"relu_{i}", nn.ReLU())
                else:
                    self.output_classifier.add_module(f"lin_{i}", nn.Linear(classifier_feature_list[i - 1], classifier_feature_list[i]))
                    self.output_classifier.add_module(f"bn_{i}", nn.BatchNorm1d(classifier_feature_list[i]))
                    self.output_classifier.add_module(f"relu_{i}", nn.ReLU())
            self.output_classifier.add_module("output", nn.Linear(classifier_feature_list[-1], output_feature))
        self.output_classifier.add_module(f"softmax", nn.Softmax())
        self.output_classifier.to(device)


    def forward(self, x):
        x_bacteria = x[:, :self.bacteria_num]
        x_virus = x[:, self.bacteria_num:self.bacteria_num + self.virus_num]
        x_non_infected = x[:, self.bacteria_num + self.virus_num:]
        # print(x_bacteria.shape)
        # print(x_virus.shape)
        # print(x_non_infected.shape)
        assert x_non_infected.shape[1] == self.noninfected_num
        x_bacteria_feature = self.linear_bacteria(x_bacteria)
        x_virus_feature = self.linear_virus(x_virus)
        x_non_infected_feature = self.linear_noninfected(x_non_infected)
        output_bacteria = self.binary_classifier_bacteria(x_bacteria_feature)
        output_virus = self.binary_classifier_virus(x_virus_feature)
        output_noninfected = self.binary_classifier_noninfected(x_non_infected_feature)


        # feature = x_bacteria_feature + x_virus_feature + x_non_infected_feature
        feature = torch.cat([x_bacteria_feature, x_virus_feature, x_non_infected_feature], 1)
        # print("feature", feature.shape)
        output = self.output_classifier(feature)
        return output, output_bacteria, output_virus, output_noninfected
    def get_algorithm_file_name(self):
        inner = self.__class__.__name__

        inner += f"_h{self.hidden_feature}"
        inner += f"_classifier_h"
        for feature in self.classifier_feature_list:
            inner += f"_{feature}"
        inner += f"_o{self.output_feature}"
        return inner
if __name__ == "__main__":
    # model = three_input_model_add(b_num=10, v_num=11, n_num=12, hidden_feature=16, classifier_feature_list=[16, 8])
    # model = three_input_model_concatenate_subnet_train(b_num=10, v_num=11, n_num=12, hidden_feature=16, classifier_feature_list=[8, 4])
    # print(model)
    # data = torch.ones(64, 33).to(device)
    # output = model(data)
    # print(len(output))
    # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, )

    # hidden
    # classifier
    #
    # hidden
    # classifier 1
    # classifier 2

    # hidden
    # classifier 1
    # classifier 2
    # classifier 3

    model = three_input_model_concatenate_subnet_train(b_num=3, v_num=4,
                                                       n_num=5,
                                                       hidden_feature=16,
                                                       classifier_feature_list=[4],
                                                       output_feature=3)
    # print(model)
    # print(model.parameters)
    print(model.parameters())

    cnt = 0
    # for param in model.parameters():
    #     cnt += 1
    #     print(cnt)
    #     print(param)
    #     param.requires_grad = False

    for child in model.children():
        cnt += 1
        print(cnt)
        print(child)
        # param.requires_grad = False