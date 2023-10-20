import torch
from collections import OrderedDict

import NeuralNetwork


class NetworkClassifier:
    def __init__(self,inputs, hidden_layers, outputs,hidden_layer_size):
        self.x = inputs
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")
        model_def = OrderedDict()
        model_def['input'] = nn.Linear(len(inputs[0])-1,hidden_layer_size,bias=False)
        nn.init.xavier_uniform_(model_def['input'].weight)
        for count in range(hidden_layers):
            model_def["relu"+str(count)] = nn.ReLU()

        self.model = NeuralNetwork(model_def).to(self.device)


    def test_loop(self, dataloader, loss_fn):
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch in range(size-2):
                X, y = dataloader.dataset.__getitem__(batch+1)
                X.requires_grad_()
                X = X.to(self.device)
                nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                pred = self.model(X)
                s = nn.Softmax(dim=0)
                pred = s(pred)
                prediction = pred.float()
                if isinstance(y,str):
                    l = self.model.conv(y)
                target = [0] * len(prediction)
                target[int(y)] = 1
                test_loss += loss_fn(prediction, torch.Tensor(target).to(self.device)).item()
                correct += (prediction.argmax(0) == int(y)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def train_loop(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        self.model.train()
        for batch in range(size-2):
            X, y = dataloader.dataset.__getitem__(batch+1)
            X.requires_grad_()
            X = X.to(self.device)
            X.retain_grad()
            nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            mean, std, var = torch.mean(X), torch.std(X), torch.var(X)
            X = (X-mean)/std
            pred = self.model(X)
            s = nn.Softmax(dim=0)
            prediction = s(pred.float())
            if isinstance(y,str):
                l = self.model.conv(y)
            target = [0] * len(prediction)
            target[int(y)] = 1
            targetT = torch.Tensor(target).to(self.device)
            loss = loss_fn(prediction, targetT)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")