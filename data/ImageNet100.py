import torch, torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np


class ImageNet100(ImageFolder):
    def __init__(self, root,
                 transform=None,
                 target_transform=None):
        super(ImageNet100, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.imgs = np.array([s[0] for s in self.samples])
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.imgs[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if len(self.TestData) == 0 else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if len(self.TestLabels) == 0 else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.imgs[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))

    def getTrainData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.imgs[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        path, target = self.TrainData[index], self.TrainLabels[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def getTestItem(self, index):
        path, target = self.TestData[index], self.TestLabels[index]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if len(self.TrainData) != 0:
            return self.getTrainItem(index)
        elif len(self.TestData) != 0:
            return self.getTestItem(index)

    def __len__(self):
        if len(self.TrainData) != 0:
            return len(self.TrainData)
        elif len(self.TestData) != 0:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.imgs[np.array(self.targets) == label]
