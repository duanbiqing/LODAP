import copy
import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import sys
import numpy as np

from DataPruning import compute_el2n_score, prune_dataset
from myNetwork import network
from data.iCIFAR100 import iCIFAR100
from data.ImageNet100 import ImageNet100
from data.TinyImagenet import TinyImageNet
from models.resnet18_ghost_no0 import resnet18_cbam0
from models.resnet18 import resnet18
from models.resnet_tiny import resnet18_tiny


def filter_para(model,  lr):
    return [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr}]

class LODAP:
    def __init__(self, args, file_name, feature_extractor, task_size, device, num_workers):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.ss_num = 2 if args.ss else 1
        self.model = network(args.fg_nc*self.ss_num, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.num_workers = num_workers
        self.old_model = None
        self.el2n_scores = []
        if args.data_name == 'cifar100':
            self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                      transforms.RandomHorizontalFlip(p=0.5),
                                                      transforms.ColorJitter(brightness=0.24705882352941178),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
            self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
            self.img_wh = 32
        elif args.data_name == 'imagenet100':
            self.train_transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.train_dataset = ImageNet100(root='../dataset/Imagenet-100/Imagenet100/train/', transform=self.train_transform)
            self.test_dataset = ImageNet100(root='../dataset/Imagenet-100/Imagenet100/val/', transform=self.test_transform)
            self.img_wh = 224
        elif args.data_name == 'tiny_imagenet':
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.train_dataset = TinyImageNet(root='../dataset/tiny-imagenet-200', train=True, transform=self.train_transform)
            self.test_dataset = TinyImageNet(root='../dataset/tiny-imagenet-200', train=False, transform=self.test_transform)
            self.img_wh = 64
        self.train_loader = None
        self.test_loader = None

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
        else:
            classes = [self.numclass-self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass*self.ss_num)

            if self.args.data_name == 'cifar100':
                backbone = resnet18_cbam0(mode='parallel_adapters').to(self.device)
            elif self.args.data_name == 'imagenet100':
                backbone = resnet18(mode='parallel_adapters').to(self.device)
            elif self.args.data_name == 'tiny_imagenet':
                backbone = resnet18_tiny(mode='parallel_adapters').to(self.device)

            model_dict = backbone.state_dict()
            para_dict = self.model.feature.state_dict()
            state_dict = {k: v for k, v in para_dict.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            backbone.load_state_dict(model_dict)
            self.model.feature = backbone
            self.model.fix_backbone_adapter()

        try:
            path = self.args.save_path + self.file_name + '/'
            filename = path + '%d_model.pkl' % self.numclass
            print(filename)
            self.model = torch.load(filename)
            print('load model:', filename)
            flag=1
        except:
            print('No %d_model' % self.numclass)
            flag=0
        self.model.train()
        self.model.to(self.device)

        return flag

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.num_workers)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 num_workers=self.num_workers)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 num_workers=self.num_workers)
        return test_loader

    def train(self, current_task, flag, old_class=0):
        if not flag:
            self.model.train()
            self.model.to(self.device)

            lr = self.learning_rate if current_task == 0 else 0.0002
            weight_decay = 0.0002 if current_task == 0 else 0.0002
            epochs = self.epochs if current_task == 0 else 61
            optim_para = filter_para(self.model, lr)
            opt = torch.optim.Adam(optim_para, weight_decay=weight_decay)
            scheduler = StepLR(opt, step_size=20, gamma=0.1)

            for epoch in range(epochs):
                if epoch == 21:
                    self.el2n_scores = compute_el2n_score(self.model, self.train_loader, self.args.ss)
                    pruned_train_dataset = prune_dataset(self.train_dataset, self.el2n_scores, self.args.prune_fraction)
                    self.train_loader = DataLoader(dataset=pruned_train_dataset, shuffle=True,
                                                 batch_size=self.args.batch_size,
                                                 num_workers=self.num_workers)
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    images, target = images.to(self.device), target.to(self.device)

                    if self.args.ss:
                        rot90k = torch.randint(1, 4, (1,))[0]
                        images = torch.stack([images, torch.rot90(images, rot90k, (2, 3))], 1)

                        images = images.view(-1, 3, self.img_wh, self.img_wh)
                        target = torch.stack([target * self.ss_num, target * self.ss_num + 1], 1).view(-1)

                    opt.zero_grad()
                    loss = self._compute_loss(images, target, old_class)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                scheduler.step()
                if epoch % self.args.print_freq == 0:
                    accuracy = self._test(self.test_loader)
                    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), end='  ')
                    print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
                    print(len(self.train_loader.dataset))



        else:
            accuracy = self._test(self.test_loader)
            print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'), end='  ')
            print('accuracy:%.5f' % accuracy)

        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0

        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            if self.args.ss:
                outputs = outputs[:, ::self.ss_num]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)

        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target.long())
        if self.old_model is None:
            return loss_cls
        else:
            feature = self.model.feature(imgs)
            with torch.no_grad():
                feature_old = self.old_model.feature(imgs)

            proto = torch.from_numpy(np.array(self.prototype)).t().to(self.device)
            proto_nor = torch.nn.functional.normalize(proto, p=2, dim=0, eps=1e-12)
            feature_nor = torch.nn.functional.normalize(feature, p=2, dim=-1, eps=1e-12)
            cos_dist = feature_nor @ proto_nor
            cos_dist = torch.max(cos_dist, dim=-1).values
            cos_dist2 = 1 - cos_dist

            loss_cls = torch.mean(loss_cls * cos_dist2, dim=0)
            loss_kd = torch.norm(feature - feature_old, p=2, dim=1)
            loss_kd = torch.sum(loss_kd * cos_dist, dim=0)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(self.class_label[index[0]]*self.ss_num)
                # proto_aug_label.append(self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label.long())

            return loss_cls + self.args.protoAug_weight*loss_protoAug + self.args.kd_weight*loss_kd

    def afterTrain(self, current_task):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

        if current_task > 0:
            model_dict = self.model.state_dict()
            for k, v in model_dict.items():
                if 'adapter' in k:
                    if 'adapter1' in k:
                        k_conv3 = k.replace('adapter1', 'primary_conv.0')
                    elif 'adapter2' in k:
                        k_conv3 = k.replace('adapter2', 'cheap_operation.0')
                    model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                    model_dict[k] = torch.zeros_like(v)
            self.model.load_state_dict(model_dict)
        elif current_task == 0:
            para_dict = self.model.state_dict()
            para_dict_re = self.structure_reorganization(para_dict)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in para_dict_re.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)


    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()

        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(loader):
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())

        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]


        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)
        print(np.shape(prototype),sys.getsizeof(prototype[0].dtype))
        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)

    def structure_reorganization(self, para_dict):
        para_dict_re = copy.deepcopy(para_dict)
        for k, v in para_dict.items():
            if 'primary_conv.1.weight' in k or 'cheap_operation.1.weight' in k:
                k_conv3 = k.replace('1.weight', '0.weight')
                k_conv3_bias = k_conv3.replace('weight', 'bias')

                k_bn_bias = k.replace('weight', 'bias')
                k_bn_mean = k.replace('weight', 'running_mean')
                k_bn_var = k.replace('weight', 'running_var')

                gamma = para_dict[k]
                beta = para_dict[k_bn_bias]
                running_mean = para_dict[k_bn_mean]
                running_var = para_dict[k_bn_var]
                eps = 1e-5
                std = (running_var + eps).sqrt()
                t = (gamma / std).reshape(-1, 1, 1, 1)
                # print(t.shape, para_dict_re[k_conv3].shape)
                para_dict_re[k_conv3] *= t
                para_dict_re[k_conv3_bias] = beta - running_mean * gamma / std
        return para_dict_re