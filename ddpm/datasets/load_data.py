import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def getUnlDevNum(unl_dev):
    if unl_dev == '':
        return []
    unl_dev_list = []
    for dev in unl_dev.split('+'):
        unl_dev_list.append(int(dev))
    return unl_dev_list


def load_data(args):
    assert args.unl_ratio < 1., 'unl_ratio should be less than 1.'

    num_classes = 10
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    unl_clses = getUnlDevNum(args.unl_cls)
    rem_clses = list(set(range(0, num_classes)) - set(unl_clses))
    # train data
    dataset_all = CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
    targets = dataset_all.targets
    idxx = [i for i, x in enumerate(targets) if x in unl_clses]
    dataset_unl = Subset(dataset_all, idxx)
    trainset_rem = Subset(dataset_all, [i for i in range(len(dataset_all)) if i not in idxx])
    # test data
    testset_all = CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=test_transform)
    test_idxx = [i for i, x in enumerate(testset_all.targets) if x in unl_clses]
    testset_unl = Subset(testset_all, test_idxx)
    testset_rem = Subset(testset_all, [i for i in range(len(testset_all)) if i not in test_idxx])
    # unlearned data
    train_length = int(len(trainset_rem))
    trainunl_length = int((args.unl_ratio * train_length) / (1 - args.unl_ratio))
    trainset_unl, _ = torch.utils.data.random_split(dataset_unl, [trainunl_length, len(dataset_unl) - trainunl_length])
    # all train data
    trainset_all = trainset_rem + trainset_unl
    # print(len(trainset_all), len(trainset_rem), len(trainset_unl), len(testset_rem))

    trainloader_all = DataLoader(trainset_all, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_rem = DataLoader(trainset_rem, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    trainloader_unl = DataLoader(trainset_unl, batch_size=args.train_batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    num_examples = {"trainset_all": len(trainset_all), "trainset_rem": len(trainset_rem), "trainset_unl": len(trainset_unl)}
    return num_classes, trainloader_all, trainloader_rem, trainloader_unl, num_examples


