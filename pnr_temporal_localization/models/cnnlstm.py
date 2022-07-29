import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class CnnLstmSys():
    def __init__(self):
        self.model = CnnLstm()
        self.loss = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4
        )


class CnnLstm(nn.Module):
    def __init__(self, hidden_size=512, num_layers=1):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        x = x.permute((0,2,1,3,4))
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.backbone(x)
        x = x.squeeze(2).squeeze(2)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        out = self.regressor(x).squeeze(2)

        return torch.sigmoid(out)