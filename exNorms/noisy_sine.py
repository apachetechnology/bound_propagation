import os
import sys
sys.path.append('.')

from argparse import ArgumentParser
from math import sqrt

import numpy as np
from tqdm import trange

import torch
from torch import distributions, nn, optim
from torch.utils.data import TensorDataset

from matplotlib import pyplot as plt

from bound_propagation import BoundModelFactory, HyperRectangle, Parallel, LpNormSet

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cpu',
                        help='Select device for tensor operations')
    parser.add_argument('--dim', choices=[1, 2], type=int, default=1, help='Dimensionality of the noisy sine')

    return parser.parse_args()

def PlotMe(X, y, y_pred):
    plt.figure(figsize=(12,5))
    plt.plot(X, y, label = 'Original')
    plt.plot(X, y_pred, label = 'Predicted')

    #plt.xticks(range(0, 360, 10))
    #plt.xticks(rotation=90)
    plt.xlabel('X')
    plt.ylabel('y')
    #plt.title('Pertubation (PGD) vs. Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

######################################################################################################################
# 
class CPlot:
    def __init__(self, aNet):
        print('Cplot')
        self.mNet = aNet

    def bound_propagation(self, input_bounds, alpha=False):
        factory = BoundModelFactory()
        bounded_model = factory.build(self.mNet)

        ibp_bounds = bounded_model.ibp(input_bounds).cpu()
        crown_bounds = bounded_model.crown(input_bounds, alpha=alpha).cpu()

        input_bounds = input_bounds.cpu()

        return input_bounds, ibp_bounds, crown_bounds


    def plot_bounds_1d(self):
        num_slices = 20

        boundaries = torch.linspace(-2, 2, num_slices + 1, device=cDevice).view(-1, 1)
        lower_x, upper_x = boundaries[:-1], boundaries[1:]
        input_bounds = HyperRectangle(lower_x, upper_x)

        input_bounds, ibp_bounds, crown_bounds = self.bound_propagation(input_bounds, alpha=True)

        plt.figure(figsize=(6.4 * 2, 3.6 * 2))
        plt.ylim(-2.5, 4)

        for i in range(num_slices):
            x1, x2 = input_bounds.lower[i].item(), input_bounds.upper[i].item()
            y1, y2 = ibp_bounds.lower[i].item(), ibp_bounds.upper[i].item()

            plt.plot([x1, x2], [y1, y1], color='blue', label='IBP lower' if i == 0 else None)
            plt.plot([x1, x2], [y2, y2], color='orange', label='IBP upper' if i == 0 else None)

            y1, y2 = crown_bounds.lower[0][i, 0, 0] * x1 + crown_bounds.lower[1][i], crown_bounds.lower[0][i, 0, 0] * x2 + crown_bounds.lower[1][i]
            y3, y4 = crown_bounds.upper[0][i, 0, 0] * x1 + crown_bounds.upper[1][i], crown_bounds.upper[0][i, 0, 0] * x2 + crown_bounds.upper[1][i]

            y1, y2 = y1.item(), y2.item()
            y3, y4 = y3.item(), y4.item()

            plt.plot([x1, x2], [y1, y2], color='darkgreen', label='CROWN lower' if i == 0 else None)
            plt.plot([x1, x2], [y3, y4], color='red', label='CROWN upper' if i == 0 else None)

        X = torch.linspace(-2, 2, 1000, device=cDevice).view(-1, 1)
        y = self.mNet(X)
        X, y = X.cpu().numpy(), y.cpu().numpy()

        plt.plot(X, y, color='black', label='Function to bound')

        plt.title(f'Bound propagation')
        plt.legend()
        plt.grid(True)
        plt.show()
        # plt.savefig(f'visualization/lbp.pdf', bbox_inches='tight', dpi=300)


    def plot_partition(self, input_bounds, ibp_bounds, crown_bounds):
        input_bounds = input_bounds.bounding_hyperrect()
        x1, x2 = input_bounds.lower, input_bounds.upper

        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 10), torch.linspace(x1[1], x2[1], 10))

        # Plot IBP
        y1, y2 = ibp_bounds.lower.item(), ibp_bounds.upper.item()
        y1, y2 = torch.full_like(x1, y1), torch.full_like(x1, y2)

        surf = ax.plot_surface(x1, x2, y1, color='blue', label='IBP', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d  # These are hax due to a bug in Matplotlib
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y2, color='blue', alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot LBP
        y_lower = crown_bounds.lower[0][0, 0] * x1 + crown_bounds.lower[0][0, 1] * x2 + crown_bounds.lower[1]
        y_upper = crown_bounds.upper[0][0, 0] * x1 + crown_bounds.upper[0][0, 1] * x2 + crown_bounds.upper[1]

        surf = ax.plot_surface(x1, x2, y_lower, color='green', label='CROWN', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        surf = ax.plot_surface(x1, x2, y_upper, color='green', alpha=0.4, shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # Plot function
        x1, x2 = input_bounds.lower, input_bounds.upper
        x1, x2 = torch.meshgrid(torch.linspace(x1[0], x2[0], 50), torch.linspace(x1[1], x2[1], 50))
        X = torch.cat(tuple(torch.dstack([x1, x2]))).to(cDevice)
        y = self.mNet(X).view(50, 50)
        y = y.cpu()

        surf = ax.plot_surface(x1, x2, y, color='red', label='Function to bound', shade=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.title(f'Bound propagation')
        plt.legend()
        plt.show()

    def plot_bounds_2d(self):
        num_slices = 20

        x_space = torch.linspace(-2.0, 2.0, num_slices + 1, device=cDevice)
        cell_width = (x_space[1] - x_space[0]) / 2
        slice_centers = (x_space[:-1] + x_space[1:]) / 2

        cell_centers = torch.cartesian_prod(slice_centers, slice_centers)
        input_bounds = HyperRectangle.from_eps(cell_centers, cell_width)

        input_bounds, ibp_bounds, crown_bounds = self.bound_propagation(input_bounds, alpha=True)

        # Plot function over entire space
        plt.clf()
        ax = plt.axes(projection='3d')

        x1, x2 = torch.meshgrid(torch.linspace(-2.0, 2.0, 500), torch.linspace(-2.0, 2.0, 500))
        X = torch.cat(tuple(torch.dstack([x1, x2]))).to(cDevice)
        y = self.mNet(X).view(500, 500)
        y = y.cpu()

        surf = ax.plot_surface(x1, x2, y, color='red', alpha=0.8)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        # General plot config
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f'Sine approximator & partitioning')
        plt.show()

        for i in trange(num_slices ** 2):
            self.plot_partition(input_bounds[i], ibp_bounds[i], crown_bounds[i])

######################################################################################################################
# 
class NoisySineDataset(TensorDataset):
    def __init__(self, dim=1, sigma=0.05, train_size=2 ** 10):
        if dim == 1:
            X, y = self.f_1d(train_size, sigma)
        elif dim == 2:
            X, y = self.f_2d(train_size, sigma)
        else:
            raise NotImplementedError()

        super().__init__(X, y)

    def noise(self, train_size, sigma):
        dist = distributions.Normal(0.0, sigma)

        train_size = (train_size,)
        return dist.sample(train_size)

    def f_1d(self, train_size, sigma):
        X = torch.linspace(-1.0, 1.0, train_size).view(-1, 1)
        return X, torch.sin(2 * np.pi * X) + self.noise(train_size, sigma).view(-1, 1)


    def f_2d(self, train_size, sigma):
        x_space = torch.linspace(-1.0, 1.0, int(sqrt(train_size)))
        X = torch.cartesian_prod(x_space, x_space)
        y = 0.5 * torch.sin(2 * np.pi * X[:, 0]) + 0.5 * torch.sin(2 * np.pi * X[:, 1]) + self.noise(train_size, sigma)

        return X, y.view(-1, 1)

######################################################################################################################
# 
class Model(nn.Sequential):
    def __init__(self, dim=1):
        super().__init__(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            Parallel(nn.ReLU(), nn.Tanh(), split_size=32),
            nn.Linear(64, 1)
        )

######################################################################################################################
# 
class CConsole:
    def __init__(self):
        print('CConsole')
        self.mNet = Model(dim=cDim).to(cDevice)

    def init(self):
        dataset = NoisySineDataset(dim=cDim, train_size=2**12)
        self.X, self.y = dataset[:]
        self.X, self.y = self.X.to(cDevice), self.y.to(cDevice)

        factory = BoundModelFactory()
        self.bounded_model = factory.build(self.mNet)

        self.optimizer = optim.Adam(self.mNet.parameters(), lr=4e-3)
        self.criterion = nn.MSELoss(reduction='none')

    def train(self, strModelName, eps=0.005):
        self.init()

        for epoch in trange(1000):
            self.optimizer.zero_grad(set_to_none=True)

            y_pred = self.mNet(self.X)
            loss = self.criterion(y_pred, self.y).mean()

            # general::ibp
            interval_bounds = self.bounded_model.ibp(HyperRectangle.from_eps(self.X, eps))
            loss = loss + torch.max(self.criterion(interval_bounds.lower, self.y), 
                                    self.criterion(interval_bounds.upper, self.y)).mean()

            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 99:
                with torch.no_grad():
                    y_pred = self.mNet(self.X)
                    loss = self.criterion(y_pred, self.y).mean().item()

                    print(f"loss: {loss:>7f}")

            torch.save({'state_dict': self.bounded_model.state_dict(), 'epoch': epoch}, strModelName)

        print('Training Done')
            

    #-----------------------------
    def evaluate(self):
        dataset = NoisySineDataset(dim=cDim)
        X_train, y_train = dataset[:]
        X, y = X_train.to(cDevice), y_train.to(cDevice)

        y_pred = self.mNet(X)

        PlotMe(X, y, y_pred)

        criterion = nn.MSELoss()
        loss = criterion(y_pred, y)

        print(f'MSE: {loss.item()}')

    @torch.no_grad()
    def plot_bounds(self):
        objPlot = CPlot(self.mNet)

        if cDim == 1:
            objPlot.plot_bounds_1d()
        elif cDim == 2:
            objPlot.plot_bounds_2d()
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def test(self, strModelPath):
        #os.makedirs('visualization', exist_ok=True)
        checkpoint = torch.load(strModelPath, map_location=torch.device(cDevice))
        self.mNet.load_state_dict(checkpoint["state_dict"])
        self.mNet.eval()

        self.evaluate()
        self.plot_bounds()

########################################################################################################################
if __name__ == '__main__':
    #args = parse_arguments()
    os.makedirs('./working_data', exist_ok=True)
    
    cDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cDim = 1 # 2 for 3D

    objCon = CConsole()

    cCommand = 'TEST' # 'TRAIN' # 
    
    fEPS = 0.01
    strModelName = './working_data/noisy_size_' + str(fEPS) + '.pth'

    if cCommand == 'TRAIN':
        objCon.train(strModelName, fEPS)
    elif cCommand == 'TEST':
        objCon.test(strModelName)
