import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class System():
    def __init__(self):
        self.model = CnnLstm()
        self.loss = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4
        )
        self.label_transform = IdentityTransform()


class CnnLstm(nn.Module):
    def __init__(self, hidden_size=512, num_layers=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[2]

        x = x.reshape(-1, x.shape[1], x.shape[3], x.shape[4])
        x = self.backbone(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        out = self.regressor(x).squeeze(2)

        return torch.sigmoid(out)


class IdentityTransform():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[1]