import numpy as np
import torch
from torch.utils import data
from utils.setup_elements import transforms_match
from collections import defaultdict
import copy

def create_task_composition(class_nums, num_tasks, fixed_order=False):
    classes_per_task = class_nums // num_tasks
    total_classes = classes_per_task * num_tasks
    label_array = np.arange(0, total_classes)
    if not fixed_order:
        np.random.shuffle(label_array)

    task_labels = []
    for tt in range(num_tasks):
        tt_offset = tt * classes_per_task
        task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
        print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
    return task_labels


def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y == i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]


def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))
    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]



class dataset_transform(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.y)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        if self.transform:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x.float(), self.y[idx]


class BalancedSampler(data.Sampler):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.labeldict = defaultdict(list)
        for idx, label in enumerate(y):
            self.labeldict[label].append(idx)
        self.labelset = set(self.y)
        self.num_classes = len(set(self.y))
        self.num_instances = batch_size // self.num_classes

    def __iter__(self):
        batch_idx_dict = defaultdict(list)
        for label in self.labelset:
            indices = copy.deepcopy(self.labeldict[label])
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True)
            np.random.shuffle(indices)
            batch_idx = []
            for idx in indices:
                batch_idx.append(idx)
                if len(batch_idx) == self.num_instances:
                    batch_idx_dict[label].append(batch_idx)
                    batch_idx = []

        avail_labels = copy.deepcopy(self.labelset)
        final_indices = []

        while len(avail_labels) >= self.num_classes:
            batch_indices = []
            for label in self.labelset:
                batch_idx = batch_idx_dict[label].pop(0)
                batch_indices.extend(batch_idx)
                if len(batch_idx_dict[label]) == 0:
                    avail_labels.remove(label)
            np.random.shuffle(batch_indices)
            final_indices.extend(batch_indices)

        return iter(final_indices)

    def __len__(self):
        return len(self.y) // self.batch_size


def setup_test_loader(test_data, params):
    test_loaders = []

    for (x_test, y_test) in test_data:
        test_dataset = dataset_transform(x_test, y_test, transform=transforms_match[params.data])
        test_loader = data.DataLoader(test_dataset, batch_size=params.test_batch, shuffle=True, num_workers=0)
        test_loaders.append(test_loader)
    return test_loaders


def shuffle_data(x, y):
    perm_inds = np.arange(0, x.shape[0])
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y


def train_val_test_split_ni(train_data, train_label, test_data, test_label, task_nums, img_size, val_size=0.1):
    train_data_rdm, train_label_rdm = shuffle_data(train_data, train_label)
    val_size = int(len(train_data_rdm) * val_size)
    val_data_rdm, val_label_rdm = train_data_rdm[:val_size], train_label_rdm[:val_size]
    train_data_rdm, train_label_rdm = train_data_rdm[val_size:], train_label_rdm[val_size:]
    test_data_rdm, test_label_rdm = shuffle_data(test_data, test_label)
    train_data_rdm_split = train_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    train_label_rdm_split = train_label_rdm.reshape(task_nums, -1)
    val_data_rdm_split = val_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    val_label_rdm_split = val_label_rdm.reshape(task_nums, -1)
    test_data_rdm_split = test_data_rdm.reshape(task_nums, -1, img_size, img_size, 3)
    test_label_rdm_split = test_label_rdm.reshape(task_nums, -1)
    return train_data_rdm_split, train_label_rdm_split, val_data_rdm_split, val_label_rdm_split, test_data_rdm_split, test_label_rdm_split
