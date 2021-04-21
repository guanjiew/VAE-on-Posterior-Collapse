from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from preprocess.med_dataset import PathMNIST
import os


# Data Loading
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


def load_data(args: dict):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    print('Checking Image Size: ', image_size)
    is_train = args.train
    assert image_size == 128, 'currently only image size of 128 supported'

    # if name.lower() == 'celeba':
    #     if is_train:
    #         print('**** Loading Training data from Celeba ****')
    #         root = os.path.join(dset_dir, 'CelebaTrain')
    #     else:
    #         print('**** Loading Testing data from Celeba ****')
    #         root = os.path.join(dset_dir, 'CelebaTest')
    #     if not os.path.exists(root):
    #         os.makedirs(root)
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(), ])
    #     data_kwargs = {'root': root, 'transform': transform}
    #     dset = CustomImageFolder
    #     data_args = dset(**data_kwargs)

    if name.lower() == 'pathmnist':
        if is_train:
            print('**** Loading Training data from Pathmnist ****')
            root = os.path.join(dset_dir, 'PathmnistTrain')
        else:
            print('**** Loading Testing data from Pathmnist ****')
            root = os.path.join(dset_dir, 'PathmnistTest')
        if not os.path.exists(root):
            os.makedirs(root)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), ])

        if is_train:
            data_args = PathMNIST(root, split='train', transform=transform, download=True)
        else:
            data_args = PathMNIST(root, split='test', transform=transform, download=True)

    else:
        raise NotImplementedError

    train_loader = DataLoader(data_args,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    print("**** Completed Loading data ****")
    return data_loader
