import random
import numpy as np
import torch

from torch.utils.data import Dataset

from . import tools


class NTUDataset(Dataset):
    def __init__(self, data, transform, class_indices=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
       :param only_label: only load label for ensemble score compute
        """

        self.data = data['x']
        self.labels = data['y']
        self.transform = transform
        self.class_indices = class_indices
        self.debug = debug

        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
#        self.load_data()
        if normalization:
            self.get_mean_map()

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        labels = self.labels[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if isinstance(data_numpy, np.ndarray):
            return torch.from_numpy(data_numpy), labels
        else:
            return data_numpy, labels

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def get_data(trn_data, tst_data, num_tasks, nc_first_task, validation, shuffle_classes, skip_tasks, class_order=None, cpertask=None):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    if class_order is None:
        num_classes = len(np.unique(trn_data['y']))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)
    print(class_order)

    # compute classes per task and num_tasks
    if cpertask is None:
        if nc_first_task is None:
            cpertask = np.array([num_classes // num_tasks] * num_tasks)
            for i in range(num_classes % num_tasks):
                cpertask[i] += 1
        else:
            assert nc_first_task < num_classes, "first task wants more classes than exist"
            remaining_classes = num_classes - nc_first_task
            assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
            cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
            for i in range(remaining_classes % (num_tasks - 1)):
                cpertask[i + 1] += 1
    else:
        cpertask = np.array(cpertask)

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}

    # ALL OR TRAIN
    filtering = np.isin(trn_data['y'], class_order)
    if filtering.sum() != len(trn_data['y']):
        trn_data['x'] = trn_data['x'][filtering]
        trn_data['y'] = np.array(trn_data['y'])[filtering]
    for this_image, this_label in zip(trn_data['x'], trn_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['trn']['x'].append(this_image)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    filtering = np.isin(tst_data['y'], class_order)
    if filtering.sum() != len(tst_data['y']):
        tst_data['x'] = tst_data['x'][filtering]
        tst_data['y'] = tst_data['y'][filtering]
    for this_image, this_label in zip(tst_data['x'], tst_data['y']):
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)
        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_sample = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_sample.sort(reverse=True)
                for ii in range(len(rnd_sample)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_sample[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_sample[ii]])
                    data[tt]['trn']['x'].pop(rnd_sample[ii])
                    data[tt]['trn']['y'].pop(rnd_sample[ii])

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ['trn', 'val', 'tst']:
            data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

    # other
    n = 0
    for t in data.keys():
        if t in skip_tasks:
            taskcla.append((t, data[t]['ncla'], False))
        else:
            taskcla.append((t, data[t]['ncla'], True))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order
