import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader, DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader, DataLoader
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from data_preprocess import CustomDataset



def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()



class GCN(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2):

        super().__init__()

        self.feature_type = feature_type
        if feature_type == 'discrete':
            self.conv1 = GCNConv(num_node_features, num_classes)

        elif feature_type == 'continuous':
            self.conv1 = GCNConv(num_node_features, 8)
            self.fc1 = torch.nn.Linear(8, 1)

    #             self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        if self.feature_type == 'discrete':
            return F.log_softmax(x, dim=1), x
        else:
            x = self.fc1(x)
            #             x = self.fc2(x)
            return x

    def fit(self, dataset, optimizer, feature_type, n_epoch=100):
        losses = []
        self.train()
        for epoch in range(1, n_epoch + 1):
            #         model.train()
            optimizer.zero_grad()
            if feature_type == 'discrete':
                #             m = nn.Sigmoid()
                #             loss = nn.BCELoss()
                #             print(dataset.y[dataset.train_mask].type())
                #             print(torch.argmax(model(dataset)[dataset.train_mask], dim=1).type())
                #             loss = loss(m(torch.argmax(model(dataset)[dataset.train_mask], dim=1)), dataset.y[dataset.train_mask].float())
                loss = F.nll_loss(self(dataset.x, dataset.edge_index)[0][dataset.train_mask],
                                  dataset.y[dataset.train_mask])

            #             loss = F.cross_entropy(model(dataset)[dataset.train_mask], dataset.y[dataset.train_mask])

            #             print(dataset.y[dataset.train_mask].shape)
            #             print(torch.argmax(model(dataset)[dataset.train_mask], dim=1).type())
            #             loss = F.mse_loss(model(dataset)[dataset.train_mask], dataset.y[dataset.train_mask])

            elif feature_type == 'continuous':
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(self(dataset.x, dataset.edge_index).flatten()[dataset.train_mask],
                                dataset.y[dataset.train_mask])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('epoch ', epoch, ' done!')

        if feature_type == 'continuous':
            plt.plot(list(range(1, n_epoch + 1)), losses)

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)
            correct = (pred.argmax(dim=1)[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            print('MSE loss for test data: ', output)
        return pred


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2):
        super().__init__()

        self.feature_type = feature_type
        if feature_type == 'discrete':
            self.sage1 = SAGEConv(num_node_features, num_classes, 'mean')

        elif feature_type == 'continuous':
            self.sage1 = SAGEConv(num_node_features, 8, 'mean')
            self.fc1 = torch.nn.Linear(8, 1)

    #             self.fc2 = torch.nn.Linear(4, 1)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        if self.feature_type == 'discrete':
            return F.log_softmax(x, dim=1), x
        else:
            x = self.fc1(x)
            #             x = self.fc2(x)
            return x

    def fit(self, dataset, optimizer, feature_type, n_epoch=200):

        # Create batches with neighbor sampling
        train_loader = NeighborLoader(
            dataset,
            num_neighbors=[5, 10],
            batch_size=16,
            input_nodes=dataset.train_mask,
        )
        losses = []
        self.train()
        for epoch in range(1, n_epoch + 1):
            total_loss = 0
            acc = 0

            for batch in train_loader:
                optimizer.zero_grad()

                if feature_type == 'discrete':
                    out, _ = self(batch.x, batch.edge_index)
                    loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                    #                     loss = torch.nn.CrossEntropyLoss()(out[batch.train_mask], batch.y[batch.train_mask])
                    acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                    batch.y[batch.train_mask])

                elif feature_type == 'continuous':
                    out = self(batch.x, batch.edge_index)
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(torch.reshape(out[batch.train_mask], (-1,)), batch.y[batch.train_mask])

                total_loss += loss
                loss.backward()
                optimizer.step()

            # if (epoch % 50 == 0):
            #     if feature_type == 'discrete':
            #         print(
            #             f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f} 'f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}%')
            #
            #     else:
            #         print(f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f}')
            losses.append(total_loss.item())
        if feature_type == 'continuous':
            plt.plot(list(range(1, n_epoch + 1)), losses)

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)
            correct = (pred.argmax(dim=1)[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            print('MSE loss for test data: ', output)
        return pred


class GIN(torch.nn.Module):
    def __init__(self, feature_type, num_node_features, num_classes=2, dim_h=32):
        super(GIN, self).__init__()
        self.feature_type = feature_type
        self.gin1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h, dim_h * 3)

        if feature_type == 'discrete':
            self.lin2 = Linear(dim_h * 3, num_classes)
        elif feature_type == 'continuous':
            self.lin2 = Linear(dim_h * 3, 1)

    def forward(self, x, edge_index):
        # Node embeddings
        h1 = self.gin1(x, edge_index)

        # Graph-level readout
        # h1 = global_add_pool(h1, batch)

        # Concatenate graph embeddings
        # h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h1)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        if self.feature_type == 'discrete':
            return F.log_softmax(h, dim=1), h
        else:
            return h

    def fit(self, dataset, optimizer, feature_type, n_epoch=300):

        # Create batches

        train_dataset = dataset.subgraph(dataset.train_mask)
        train_dataset = CustomDataset.from_data(train_dataset, dataset.input_feature_type, dataset.target_feature_type)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        self.train()
        for epoch in range(n_epoch + 1):
            total_loss = 0
            acc = 0

            for batch in train_loader:
                optimizer.zero_grad()

                if feature_type == 'discrete':
                    out = self(batch.x, batch.edge_index.T)
                    loss = F.nll_loss(out[0][batch.train_mask], batch.y[batch.train_mask])
                    acc += accuracy(out[0][batch.train_mask].argmax(dim=1),
                                    batch.y[batch.train_mask])

                elif feature_type == 'continuous':
                    out = self(batch.x, batch.edge_index.T)
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(torch.reshape(out[batch.train_mask], (-1,)), batch.y[batch.train_mask])

                total_loss += loss

                loss.backward()
                optimizer.step()

            # if(epoch % 50 == 0):
            #    if feature_type == 'discrete':
            #        print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} 'f'| Train Acc: {acc/len(train_loader)*100:>6.2f}%')
            #
            #    else:
            #       print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f}')

    def test(self, dataset, feature_type):
        self.eval()
        if feature_type == 'discrete':
            pred, _ = self(dataset.x, dataset.edge_index)[0].argmax(dim=1)
            correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
        elif feature_type == 'continuous':
            pred = self(dataset.x, dataset.edge_index).flatten()
            mse_loss = torch.nn.MSELoss()
            output = mse_loss(pred.flatten()[dataset.test_mask], dataset.y[dataset.test_mask])
            # print('MSE loss for test data: ', output)
        return pred