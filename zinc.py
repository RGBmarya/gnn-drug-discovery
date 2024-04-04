import argparse
import os.path as osp
import time
from torch_geometric.datasets import ZINC
from torchmetrics.regression import MeanSquaredError, R2Score
from torch.nn.functional import relu
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.nn import global_mean_pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
#            hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'TUDataset')
train_dataset = ZINC(path, subset=True, split="train")
train_dataset.to(device)
test_dataset = ZINC(path, subset=True, split="test")
train_dataset.to(device)



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads)
        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1,
                               concat=False, dropout=0.6)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(0.6)  # Increase or decrease based on your dataset size and complexity

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = relu(x)
        x = self.bn1(x)
        x = self.dropout(x)  # Apply dropout after batch normalization and activation

        x = self.conv2(x, edge_index)
        x = relu(x)
        x = self.bn2(x)
        x = self.dropout(x)  # Apply dropout again for the next layer

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


# `train_dataset.num_classes)` is 9994 instead of 1 for some reason (probably user error but just
# hard-coded 1 for out_channels for simplicity)
model = GAT(train_dataset.num_node_features, args.hidden_channels, 1,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()
mean_squared_error = MeanSquaredError()
r2score = R2Score()


def train():
    model.train()
    mse = 0
    r2 = 0
    count = 0
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch).squeeze()
        loss = loss_fn(out, data.y)
        mse += mean_squared_error(out, data.y)
        r2 += r2score(out, data.y)
        count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (float(loss), mse/count, r2/count)


@torch.no_grad()

def test():
    return test_helper(test_loader)

def test_helper(loader):
    model.eval()
    
    mse = 0 
    r2 = 0
    count = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch).squeeze()
        mse += mean_squared_error(out, data.y)
        r2 += r2score(out, data.y)
        count += 1

    return (mse/count, r2/count)


times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss, train_mse, train_r2 = train()
    test_mse, test_r2 = test()
    # log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    print(f"Epoch={epoch}", f"Loss={loss:.4f}",
    f"MSE (train)={train_mse.item():.2f}", f"R2 (train)={train_r2.item():.2f}",
        f"MSE (test)={test_mse.item():.2f}", f"R2 (test)={test_r2.item():.2f}", sep=" | ")
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")