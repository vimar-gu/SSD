import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.convnet import ConvNet
from .buffer_utils import random_retrieve
from .augment import DiffAug


def condense_retrieve(buffer, num_samples, excl_labels=None, incl_labels=None):
    '''Retrieve condensed images from the memory
    Defaultly excluded labels are given to avoid retrieving the current
    summarizing samples. But can also retrieve based on included labels.
    '''
    if excl_labels is not None:
        avail_indices = []
        for lab in list(set(buffer.condense_dict.keys()) - set(excl_labels)):
            avail_indices += buffer.condense_dict[lab]
        num_samples = min(len(avail_indices), num_samples)
        sel_indices = np.random.choice(avail_indices, num_samples, replace=False)
    else:
        assert len(incl_labels) % num_samples == 0
        sel_indices = []
        for lab in incl_labels:
            sel_indices += list(np.random.choice(buffer.condense_dict[lab], num_samples // len(incl_labels), replace=False))

    return buffer.buffer_img[sel_indices], buffer.buffer_label[sel_indices]


def dist(x, y, metric='mse'):
    '''Distance calculation functions
    '''
    if metric == 'mse':
        return (x - y).pow(2).sum()
    elif metric == 'l1':
        return (x - y).abs().sum()
    elif metric == 'l1_mean':
        n_b = x.shape[0]
        return (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif metric == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        return torch.sum(1 - torch.sum(x * y, dim=-1) /
                         (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))


class SummarizeUpdate(object):
    def __init__(self, params):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.params = params
        self.label_dict = {}
        self.optim_flag = True

    def new_task(self, num_classes, labels):
        '''Initialize the label dict for model training
        '''
        if self.params.data == 'mini_imagenet':
            im_size = (84, 84)
        elif self.params.data == 'tiny_imagenet':
            im_size = (64, 64)
        else:
            im_size = (32, 32)
        self.model = ConvNet(num_classes, im_size=im_size).cuda()
        self.optimizer_model = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for idx, label in enumerate(labels):
            self.label_dict[label] = idx + num_classes - len(labels)
        self.new_labels = labels
        self.optim_flag = True

    def update(self, buffer, x, y, aff_x=None, aff_y=None, update_index=-1, transform=None, **kwargs):
        condense_flag = True

        if len(aff_x) < self.params.queue_size:
            condense_flag = False
        elif len(aff_x) > 0:
            aff_x = torch.cat(aff_x, dim=0)
            aff_y = torch.cat(aff_y)

        for data, label in zip(x, y):
            label_c = label.item()
            # The buffer is not full
            if self.params.mem_size > buffer.current_index:
                # The condense dict is not full
                if len(buffer.condense_dict[label_c]) < buffer.images_per_class:
                    buffer.buffer_img[buffer.current_index].data.copy_(data)
                    buffer.buffer_img_rep[buffer.current_index].data.copy_(data)
                    buffer.buffer_label[buffer.current_index].data.copy_(label)
                    buffer.condense_dict[label_c].append(buffer.current_index)
                    buffer.avail_indices.remove(buffer.current_index)
                    buffer.current_index += 1
                    condense_flag = False
                # The condense dict is full
                else:
                    buffer.buffer_img[buffer.current_index].data.copy_(data)
                    buffer.buffer_label[buffer.current_index].data.copy_(label)
                    buffer.current_index += 1
            # The buffer is full
            else:
                # The condense dict is not full
                if len(buffer.condense_dict[label_c]) < buffer.images_per_class:
                    # Choose a random index
                    replace_index = np.random.choice(buffer.avail_indices)
                    # Remove the random sample record
                    buffer.buffer_img[replace_index].data.copy_(data)
                    buffer.buffer_img_rep[replace_index].data.copy_(data)
                    buffer.buffer_label[replace_index].data.copy_(label)
                    buffer.condense_dict[label_c].append(replace_index)
                    buffer.avail_indices.remove(replace_index)
                    condense_flag = False
                # The condense dict is full -> original reservoir update
                else:
                    random_index = int(np.random.uniform(buffer.n_seen_so_far))
                    if random_index < self.params.mem_size and random_index in buffer.avail_indices:
                        buffer.buffer_img[random_index].data.copy_(data)
                        buffer.buffer_label[random_index].data.copy_(label)

        buffer.n_seen_so_far += x.shape[0]

        if update_index % self.params.summarize_interval != 0:
            condense_flag = False

        # conduct sample condense
        if condense_flag:
            labelset = set(y.cpu().numpy())
            # initialize the optimization target at the first iteration of new tasks
            if self.optim_flag:
                self.condense_x = [buffer.buffer_img[buffer.condense_dict[c]] for c in labelset]
                self.condense_x = copy.deepcopy(torch.cat(self.condense_x, dim=0)).requires_grad_()
                self.condense_y = [buffer.buffer_label[buffer.condense_dict[c]] for c in labelset]
                self.condense_y = torch.cat(self.condense_y)
                self.optimizer_img = torch.optim.SGD([self.condense_x,], lr=self.params.lr_img, momentum=0.9)
                self.optim_flag = False

            diff_aug = DiffAug(strategy='color_crop', batch=False)
            match_aug = transforms.Compose([diff_aug])

            avg_loss = 0.
            for cls_idx, c in enumerate(labelset):
                # obtain samples of each class and add augmentation
                img_real = aff_x[aff_y == c]
                lab_real = aff_y[aff_y == c]
                lab_real = torch.tensor([self.label_dict[l_real.item()] for l_real in lab_real]).cuda()
                img_syn = self.condense_x[self.condense_y == c]
                lab_syn = self.condense_y[self.condense_y == c]
                lab_syn = torch.tensor([self.label_dict[l_real.item()] for l_real in lab_syn]).cuda()

                img_aug = match_aug(torch.cat((img_real, img_syn), dim=0))
                img_real = img_aug[:len(img_real)]
                img_syn = img_aug[len(img_real):]

                # calculate matching loss
                loss = self.match_loss(img_real, img_syn, lab_real, lab_syn, buffer)
                avg_loss += loss.item()

                self.optimizer_img.zero_grad()
                loss.backward()
                self.optimizer_img.step()

                # update the condensed image to the memory
                img_new = self.condense_x[self.condense_y == c]
                buffer.buffer_img[buffer.condense_dict[c]] = img_new.detach()

        # update the matching model
        y = torch.tensor([self.label_dict[lab.item()] for lab in y]).cuda()
        self.retrieve_update_model(x, y, buffer, transform)

    def match_loss(self, img_real, img_syn, lab_real, lab_syn, buffer):
        loss = 0.

        # check if memory-based matching loss is applicable
        output_real = self.model(img_real)
        with torch.no_grad():
            feat_real = self.model.features(img_real)
        output_syn, feat_syn = self.model(img_syn, return_features=True)
        img_mem, lab_mem = condense_retrieve(buffer, self.params.mem_sim_num, excl_labels=self.new_labels)
        if len(img_mem) > 0:
            with torch.no_grad():
                feat_mem = self.model.features(img_mem)
            feat_syn = F.normalize(feat_syn, dim=1)
            feat_mem = F.normalize(feat_mem, dim=1)
            feat_real = F.normalize(feat_real, dim=1)
            mem_loss = dist(
                self.euclidean_dist(feat_syn.mean(0, keepdim=True), feat_mem),
                self.euclidean_dist(feat_real.mean(0, keepdim=True), feat_mem),
                'l1'
            )
        else:
            mem_loss = 0

        # options of feature distribution matching and gradient matching
        if 'feat' in self.params.match:
            loss += dist(feat_real.mean(0), feat_syn.mean(0), self.params.metric)
        if 'grad' in self.params.match:
            loss_real = self.criterion(output_real, lab_real)
            gw_real = torch.autograd.grad(loss_real, self.model.parameters())
            gw_real = list((_.detach().clone() for _ in gw_real))

            loss_syn = self.criterion(output_syn, lab_syn)
            gw_syn = torch.autograd.grad(loss_syn, self.model.parameters(), create_graph=True)

            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                if len(gwr.shape) == 1 or len(gwr.shape) == 2:
                    continue
                loss += dist(gwr, gws, self.params.metric)
        if 'feat' in self.params.match and 'grad' in self.params.match:
            loss = loss / 2.0

        if mem_loss > 0:
            if self.params.mem_extra == 1:
                loss = loss + mem_loss * self.params.mem_weight
            else:
                loss = loss * (1 - self.params.mem_weight) + mem_loss * self.params.mem_weight

        return loss

    def update_model(self, x, y, transform):
        '''Naive model updating
        '''
        data_len = len(x)
        for tmp_idx in range(data_len // 10):
            batch_x = x[tmp_idx * 10 : (tmp_idx + 1) * 10]
            batch_y = y[tmp_idx * 10 : (tmp_idx + 1) * 10].cuda()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()

    def retrieve_update_model(self, x, y, buffer, transform):
        '''Model updating with images of previous tasks
        '''
        labels = list(set(y.cpu().numpy()))
        condense_indices = []
        for lab in buffer.condense_dict.keys():
            condense_indices += buffer.condense_dict[lab]
        r_x, r_y = random_retrieve(buffer, 10, excl_indices=condense_indices)
        r_y = torch.tensor([self.label_dict[lab.item()] for lab in r_y]).cuda()
        r_x = torch.cat((x, r_x), dim=0)
        r_y = torch.cat((y, r_y)).long()
        random_indices = torch.randperm(len(r_x))
        r_x = r_x[random_indices]
        r_y = r_y[random_indices]
        data_len = len(r_x)
        bs = 10
        for tmp_idx in range(data_len // bs):
            batch_x = r_x[tmp_idx * bs : (tmp_idx + 1) * bs]
            batch_y = r_y[tmp_idx * bs : (tmp_idx + 1) * bs]
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            self.optimizer_model.zero_grad()
            loss.backward()
            self.optimizer_model.step()

    def euclidean_dist(self, x, y):
        m, n = x.shape[0], y.shape[0]
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        return dist
