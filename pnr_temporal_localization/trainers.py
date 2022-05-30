

class Trainer():
    def __init__(self, model, model_save_path, loss_fn, optimizer):
        self.model = model
        self.model_save_path = model_save_path
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_model(self, train_dataloader, val_dataloader, logging=True):
        if logging:
            if os.path.exists('./logs'):
                shutil.rmtree('./logs')
            writer = SummaryWriter(log_dir='./logs')

        for t in range(self.epochs):
            print(f"Epoch{t}\n--------------")
            self._train(train_dataloader)
            self._validate(val_dataloader)

            if logging:
                writer.add_scalars('Loss',
                                   {'train': self.train_loss,
                                   'val': self.val_loss},
                                   t)
                writer.add_scalars('Accuracy',
                                   {'train': self.train_acc,
                                   'val': self.val_acc},
                                   t)

        if logging:
            writer.close()

        torch.save(self.model.state_dict(), self.model_save_path)

    def _train(self, dataloader, show=True):
        self.model.train()
        for input, label in dataloader:
            pred = self.model(input)
            loss = self.loss_fn(pred, label)
            self.train_loss += loss.item()
            self.train_acc += (pred.argmax(1) == label
                                ).type(torch.float).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.train_loss /= len(dataloader)
        self.train_acc /= len(dataloader.dataset)

        if show:
            print(f"Train Error: \n Accuracy: {(100*self.train_acc):>0.1f}%, ",
                  f"Avg loss: {self.train_loss:>8f}")

    def _validate(self, dataloader, show=True):
        self.model.eval()
        with torch.no_grad():
            for input, label in dataloader:
                pred = self.model(input)
                self.val_loss += self.loss_fn(pred, label).item()
                self.val_acc += (pred.argmax(1) == label
                                    ).type(torch.float).sum().item()

        self.val_loss /= len(dataloader)
        self.val_acc /= len(dataloader.dataset)

        if show:
            print(f"Val Error: \n Accuracy: {(100*self.val_acc):>0.1f}%, ",
                  f"Avg loss: {self.val_loss:>8f}")
