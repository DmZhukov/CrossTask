from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import torch as th
from torch.utils.data import Dataset
import math

def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path,'r') as f:
        idx = f.readline()
        while idx is not '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}

def get_vids(path):
    task_vids = {}
    with open(path,'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids

def read_assignment(T, K, path):
    Y = np.zeros([T, K], dtype=np.uint8)
    with open(path,'r') as f:
        for line in f:
            step,start,end = line.strip().split(',')
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1
            Y[start:end,step] = 1
    return Y

def random_split(task_vids, test_tasks, n_train):
    train_vids = {}
    test_vids = {}
    for task,vids in task_vids.items():
        if task in test_tasks and len(vids) > n_train:
            train_vids[task] = np.random.choice(vids,n_train,replace=False).tolist()
            test_vids[task] = [vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids

def get_A(task_steps, share="words"):
    """Step-to-component matrices."""
    if share == 'words':
        # share words
        task_step_comps = {task: [step.split(' ') for step in steps] for task,steps in task_steps.items()}
    elif share == 'task_words':
        # share words within same task
        task_step_comps = {task: [[task+'_'+tok for tok in step.split(' ')] for step in steps] for task,steps in task_steps.items()}
    elif share == 'steps':
        # share whole step descriptions
        task_step_comps = {task: [[step] for step in steps] for task,steps in task_steps.items()}
    else:
        # no sharing
        task_step_comps = {task: [[task+'_'+step] for step in steps] for task,steps in task_steps.items()}
    vocab = []
    for task,steps in task_step_comps.items():
        for step in steps:
            vocab.extend(step)
    vocab = {comp: m for m,comp in enumerate(set(vocab))}
    M = len(vocab)
    A = {}
    for task,steps in task_step_comps.items():
        K = len(steps)
        a = th.zeros(M, K)
        for k,step in enumerate(steps):
            a[[vocab[comp] for comp in step],k] = 1
        a /= a.sum(dim=0)
        A[task] = a
    return A, M

class CrossTaskDataset(Dataset):
    def __init__(self, task_vids, n_steps, features_path, constraints_path):
        super(CrossTaskDataset, self).__init__()
        self.vids = []
        for task,vids in task_vids.items():
            self.vids.extend([(task,vid) for vid in vids])
        self.n_steps = n_steps
        self.features_path = features_path
        self.constraints_path = constraints_path

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        task,vid = self.vids[idx]
        X = th.tensor(np.load(os.path.join(self.features_path,vid+'.npy')), dtype=th.float)
        cnst_path = os.path.join(self.constraints_path,task+'_'+vid+'.csv')
        C = th.tensor(1-read_assignment(X.size()[0], self.n_steps[task], cnst_path), dtype=th.float)
        return {'vid': vid, 'task': task, 'X': X, 'C': C}

