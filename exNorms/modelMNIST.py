import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing
from tqdm import trange

import torchvision.datasets as datasets
import torchvision.transforms as V2

from bound_propagation import BoundModelFactory, HyperRectangle

from torchsummary import summary

####################################################################
#
class MNISTNetwork(nn.Sequential):
    def __init__(self, img_size, classes):
        
        super().__init__(
            nn.Linear(img_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, classes)
        )


#########################################################################
# 
class CClassifier:
    def __init__(self):
        print('CClassifier')
        self.mInch = 1
        self.mInDim = 28
        self.mClasses = 10

        net = MNISTNetwork(self.mInDim * self.mInDim, self.mClasses)
        factory = BoundModelFactory()
        self.mModel = factory.build(net)
        summary(self.mModel, (self.mInch, self.mInDim * self.mInDim))

    def GetMNISTDS(self, strDSPath, nBatch_size=100):
        train_data = datasets.MNIST(strDSPath, train=True, 
                                    download=False, transform=V2.ToTensor())
        test_data = datasets.MNIST(strDSPath, train=False, 
                                    download=False, transform=V2.ToTensor())
        
        self.mDLTrain = torch.utils.data.DataLoader(train_data, 
                                                      batch_size=nBatch_size, 
                                                      shuffle=True, pin_memory=True, 
                                                      num_workers=min(multiprocessing.cpu_count(),4))
        self.mDLTest = torch.utils.data.DataLoader(test_data, 
                                                     #batch_size=cBatch_size,
                                                     batch_size=1,
                                                     shuffle=True, 
                                                     pin_memory=True, 
                                                     num_workers=min(multiprocessing.cpu_count(),4))
        self.mDLTrain.mean = self.mDLTest.mean = torch.tensor([0.0])
        self.mDLTrain.std = self.mDLTest.std = torch.tensor([1.0])

        return self.mDLTrain, self.mDLTest
    
    def Train(self, strModelName):
        print('[TRAINING]')

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.mModel.parameters(), lr=5e-4)

        k = 1.0
        for epoch in trange(10):
            running_loss = 0.0
            running_cross_entropy = 0.0
            for i, (X, y) in enumerate(self.mDLTrain):
                X, y = X.to(cDevice), y.to(cDevice)
                optimizer.zero_grad(set_to_none=True)

                y_hat = self.mModel(X)

                cross_entropy = criterion(y_hat, y)

                bounds = self.mModel.ibp(HyperRectangle.from_eps(X, 0.01))
                logit = self.adversarial_logit(bounds, y)

                loss = k * cross_entropy + (1 - k) * criterion(logit, y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_cross_entropy += cross_entropy.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:3d}] loss: {running_loss / 100:.3f}, cross entropy: {running_cross_entropy / 100:.3f}')
                    running_loss = 0.0
                    running_cross_entropy = 0.0

            k = max(k - 0.1, 0.5)
            torch.save({'state_dict': self.bounded_model.state_dict(), 'epoch': epoch}, strModelName)
            # END Dataloader
        #END Epoch
        print('Training Done')


#########################################################################
if __name__ == '__main__':
    os.makedirs('./working_data', exist_ok=True)
    cDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    objClr = CClassifier()
    objClr.GetMNISTDS('../DATA')
    strModelName = './working_data/MNIST.pth'
    objClr.Train(strModelName)