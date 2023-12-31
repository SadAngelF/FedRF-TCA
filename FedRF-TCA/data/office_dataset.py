
import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

def get_pair_dataset(source, target, train_source_transform, val_transform, dataset='office31', train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    if dataset == 'office31':
        root = "/data/dataset/office31/"
        def concat_dataset(root: str, tasks: str,  **kwargs):
            return office31(root=root, task=tasks, **kwargs)
    elif dataset == 'office_caltech_10':
        root = "/data/dataset/office_caltech_10/"
        def concat_dataset(root: str, tasks: str,  **kwargs):
            return office_caltech_10(root=root, task=tasks, **kwargs)
    elif dataset == 'digit_five':
        root = "/data/dataset/digit_five/"
        def concat_dataset(root: str, tasks: str,  **kwargs):
            return digit_five_two(root=root, task=tasks, **kwargs)

    train_source_dataset = concat_dataset(root=root, tasks=source, transform=train_source_transform)
    train_target_dataset = concat_dataset(root=root, tasks=target, transform=train_target_transform)
    # val_dataset = concat_dataset(root=root, tasks=target, transform=val_transform)
    # test_dataset = val_dataset
    
    # class_names = train_source_dataset.datasets[0].classes
    class_names = train_source_dataset.classes
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, num_classes, class_names


def get_office_10_dataset(source, train_source_transform):
    # if train_target_transform is None:
    root = "/data/dataset/office_caltech_10/"
    def concat_dataset(root: str, tasks: str,  **kwargs):
        return office_caltech_10(root=root, task=tasks, **kwargs)

    train_source_dataset = concat_dataset(root=root, tasks=source, transform=train_source_transform)
    class_names = train_source_dataset.classes
    num_classes = len(class_names)

    return train_source_dataset

def get_digit_five_dataset(source, train_source_transform):
    # if train_target_transform is None:
    train_target_transform = train_source_transform

    root = "/data/dataset/digit_five/"
    def concat_dataset(root: str, tasks: str,  **kwargs):
        return digit_five_two(root=root, task=tasks, **kwargs)

    train_source_dataset = concat_dataset(root=root, tasks=source, transform=train_source_transform)
    # train_target_dataset = concat_dataset(root=root, tasks=target, transform=train_target_transform)
    # val_dataset = concat_dataset(root=root, tasks=target, transform=val_transform)
    # test_dataset = val_dataset
    
    # class_names = train_source_dataset.datasets[0].classes
    class_names = train_source_dataset.classes
    num_classes = len(class_names)

    return train_source_dataset



####################################################################

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.samples)

    def parse_data_file(self, file_name):
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self):
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented


####################################################################
class digit_five_two(ImageList):
    image_list = {
        # "train": "train/image_list.txt",
        "mnist-m": "mnist_m_train_labels.txt",
        "synthetic_digits": "digit_five_image_list_synthetic_digits.txt"
    }
    CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(self, root: str, task: str,  **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        if task == "mnist-m":
            domain_file = os.path.join(root, "mnist_m_train")
        else:
            domain_file = os.path.join(root, task)

        super(digit_five_two, self).__init__(domain_file, office_caltech_10.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
    
class office_caltech_10(ImageList):
    image_list = {
        # "train": "train/image_list.txt",
        "amazon": "office_caltech_10_image_list_amazon.txt",
        "caltech": "office_caltech_10_image_list_caltech.txt",
        "dslr": "office_caltech_10_image_list_dslr.txt",
        "webcam": "office_caltech_10_image_list_webcam.txt" 
    }
    CLASSES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

    def __init__(self, root: str, task: str,  **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        domain_file = os.path.join(root, task)

        super(office_caltech_10, self).__init__(domain_file, office_caltech_10.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

class office31(ImageList):
    image_list = {
        # "train": "train/image_list.txt",
        "amazon": "image_list_amazon.txt",
        "dslr": "image_list_dslr.txt",
        "webcam": "image_list_webcam.txt" 
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desktop_computer', 'desk_chair', 'desk_lamp', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str,  **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        domain_file = os.path.join(root, task)

        super(office31, self).__init__(domain_file, office31.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

##################################################################
if __name__ == '__main__':
    root = "/data/dataset/office31/"
    source = "amazon" # "Synthetic"
    target = "webcam" # "Real"

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]) # transforms.ToPILImage(), 
    train_transform = transform

    source_dataset, target_dataset, num_classes, class_names = \
        get_pair_dataset(root, source, target, train_transform, val_transform=train_transform)
    
    for i, (data, label) in enumerate(source_dataset):
        if i == 0:
            feature_dim = data.size()
            label_dim = label
        else:
            if feature_dim != data.size():
                print('feature_dim wrong')
    print('i = ', i)
    print('feature_dim = ', feature_dim)
    print('type(label) = ', type(label))