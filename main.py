import random

import numpy as np
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader


from method import LODAP
from models.resnet18_ghost_no0 import resnet18_cbam0
from models.resnet18 import resnet18
from models.resnet_tiny import resnet18_tiny
from dataset.iCIFAR100 import iCIFAR100
from dataset.ImageNet100 import ImageNet100
from dataset.TinyImagenet import TinyImageNet

import argparse
import torch

parser = argparse.ArgumentParser(description='Efficient Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=5, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=1.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
parser.add_argument('--ss', default=True, type=bool, help='Use self-supervision')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--prune_fraction', default=0.0, type=float, help='the ratio of pruned weights')

args = parser.parse_args()
print(args)

seed = 2025
print("Set seed", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    print()

    task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '-' + str(task_size)
    ss_num = 2 if args.ss else 1

    if args.data_name == 'cifar100':
        feature_extractor = resnet18_cbam0()
    elif args.data_name == 'imagenet100':
        feature_extractor = resnet18()
    elif args.data_name == 'tiny_imagenet':
        feature_extractor = resnet18_tiny()

    model = LODAP(args, file_name, feature_extractor, task_size, device, args.num_workers)
    class_set = list(range(args.total_nc))

    for i in range(args.task_num+1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])
        load_model_flag = model.beforeTrain(i)
        model.train(i, load_model_flag, old_class=old_class)
        model.afterTrain(i)


    ####### Test ######
    print("############# Test for each Task #############")
    if args.data_name == 'cifar100':
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    elif args.data_name == 'imagenet100':
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        test_dataset = ImageNet100(root='../dataset/Imagenet-100/Imagenet100/val/', transform=test_transform)
    elif args.data_name == 'tiny_imagenet':
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        test_dataset = TinyImageNet(root='../dataset/tiny-imagenet-200', train=False, transform=test_transform)
    acc_all = []
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task+1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                if args.ss:
                    outputs = outputs[:, ::ss_num]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print(acc_up2now)

    print(acc_all)

    print("############# Test for up2now Task #############")
    tast_all=[]
    if args.data_name == 'cifar100':
        test_dataset = iCIFAR100('./dataset', test_transform=test_transform, train=False, download=True)
    elif args.data_name == 'imagenet100':
        test_dataset = ImageNet100(root='../dataset/Imagenet-100/Imagenet100/val/', transform=test_transform)
    elif args.data_name == 'tiny_imagenet':
        test_dataset = TinyImageNet(root='../dataset/tiny-imagenet-200', train=False, transform=test_transform)
    for current_task in range(args.task_num+1):
        class_index = args.fg_nc + current_task*task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            if args.ss:
                outputs = outputs[:, ::ss_num]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        tast_all.append(accuracy)
        print('accuracy:', accuracy)
    print('average:', sum(tast_all)/len(tast_all))


if __name__ == "__main__":
    main()