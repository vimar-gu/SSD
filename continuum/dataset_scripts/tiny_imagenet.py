import pickle
import numpy as np
from continuum.data_utils import create_task_composition, load_task_with_labels, shuffle_data
from continuum.dataset_scripts.dataset_base import DatasetBase
from continuum.non_stationary import construct_ns_multiple_wrapper, test_ns

TEST_SPLIT = 1 / 6


class TinyImageNet(DatasetBase):
    def __init__(self, scenario, params):
        dataset = 'tiny_imagenet'
        if scenario == 'ni':
            num_tasks = len(params.ns_factor)
        else:
            num_tasks = params.num_tasks
        super(TinyImageNet, self).__init__(dataset, scenario, num_tasks, params.num_runs, params)

    def download_load(self):
        train_dir = './datasets/tiny-imagenet-200/train.pkl'
        test_dir = './datasets/tiny-imagenet-200/val.pkl'

        train = pickle.load(open(train_dir, 'rb'))
        self.train_data = train['data'].reshape((100000, 64, 64, 3))
        self.train_label = train['target']

        test = pickle.load(open(test_dir, 'rb'))
        self.test_data = test['data'].reshape((10000, 64, 64, 3))
        self.test_label = test['target']

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        if self.scenario == 'ni':
            x_train, y_train = self.train_set[cur_task]
            labels = set(y_train)
        elif self.scenario == 'nc':
            labels = self.task_labels[cur_task]
            x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        else:
            raise Exception('unrecognized scenario')
        return x_train, y_train, labels

    def setup(self):
        if self.scenario == 'ni':
            self.train_set, self.val_set, self.test_set = construct_ns_multiple_wrapper(self.train_data,
                                                                                        self.train_label,
                                                                                        self.test_data, self.test_label,
                                                                                        self.task_nums, 84,
                                                                                        self.params.val_size,
                                                                                        self.params.ns_type, self.params.ns_factor,
                                                                                        plot=self.params.plot_sample)

        elif self.scenario == 'nc':
            self.task_labels = create_task_composition(class_nums=200, num_tasks=self.task_nums,
                                                       fixed_order=self.params.fix_order)
            self.test_set = []
            for labels in self.task_labels:
                x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
                self.test_set.append((x_test, y_test))

    def test_plot(self):
        test_ns(self.train_data[:10], self.train_label[:10], self.params.ns_type,
                self.params.ns_factor)
