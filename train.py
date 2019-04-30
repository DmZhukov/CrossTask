from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from model import Model
from data import *
from args import parse_args
from dp import dp
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Loss(nn.Module):
    def __init__(self, lambd):
        super(Loss, self).__init__()
        self.lambd = lambd
        self.lsm = nn.LogSoftmax(dim=1)
        
    def forward(self, O, Y, C):
        return (Y*(self.lambd * C - self.lsm(O))).mean(dim=0).sum()

def uniform_assignment(T,K):
    stepsize = float(T) / K
    y = th.zeros(T,K)
    for k in range(K):
        t = round(stepsize*(k+0.5))
        y[t,k] = 1
    return y

def get_recalls(Y_true, Y_pred):
    step_match = {task: 0 for task in Y_true.keys()}
    step_total = {task: 0 for task in Y_true.keys()}
    for task,ys_true in Y_true.items():
        ys_pred = Y_pred[task]
        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            step_total[task] += (y_true.sum(axis=0)>0).sum()
            step_match[task] += (y_true*y_pred).sum()
    recalls = {task: step_match[task] / n for task,n in step_total.items()}
    return recalls

args = parse_args()

task_vids = get_vids(args.video_csv_path)
val_vids = get_vids(args.val_csv_path)
task_vids = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for task,vids in task_vids.items()}

primary_info = read_task_info(args.primary_path)
test_tasks = set(primary_info['steps'].keys())
if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info['steps'], **related_info['steps']}
    n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
else:
    task_steps = primary_info['steps']
    n_steps = primary_info['n_steps']
all_tasks = set(n_steps.keys())
task_vids = {task: vids for task,vids in task_vids.items() if task in all_tasks}

A, M = get_A(task_steps, share=args.share)

if args.use_gpu:
    A = {task: a.cuda() for task, a in A.items()}


train_vids, test_vids = random_split(task_vids, test_tasks, args.n_train)

trainset = CrossTaskDataset(train_vids, n_steps, args.features_path, args.constraints_path)
trainloader = DataLoader(trainset, 
    batch_size = args.batch_size, 
    num_workers = args.num_workers, 
    shuffle = True, 
    drop_last = True,
    collate_fn = lambda batch: batch,
    )
testset = CrossTaskDataset(test_vids, n_steps, args.features_path, args.constraints_path)
testloader = DataLoader(testset, 
    batch_size = args.batch_size, 
    num_workers = args.num_workers, 
    shuffle = False, 
    drop_last = False,
    collate_fn = lambda batch: batch,
    )

net = Model(args.d, M, A, args.q).cuda() if args.use_gpu else Model(args.d, M, A, args.q)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
loss_fn = Loss(args.lambd)

# initialize with uniform step assignment
Y = {}
for batch in trainloader:
    for sample in batch:
        task = sample['task']
        vid = sample['vid']
        K = n_steps[task]
        T = sample['X'].shape[0]
        if task not in Y:
            Y[task] = {}
        y = uniform_assignment(T,K)
        Y[task][vid] = y.cuda() if args.use_gpu else y

def train_epoch(pretrain=False):
    cumloss = 0.
    for batch in trainloader:
        for sample in batch:
            vid = sample['vid']
            task = sample['task']
            X = sample['X'].cuda() if args.use_gpu else sample['X']
            C = sample['C'].cuda() if args.use_gpu else sample['C']
            if pretrain:
                # picking random assignment, that satisfies the constraints
                O = np.random.rand(X.size()[0],n_steps[task]) + C.cpu().numpy()
                y = np.zeros(Y[task][vid].shape,dtype=np.float32)
                dp(y,O.astype(np.float32),exactly_one=True)
                Y[task][vid].data = th.tensor(y,dtype=th.float).cuda() if args.use_gpu else th.tensor(y,dtype=th.float)
            else:
                # updating assignment
                O = net(X, task)
                # y = th.tensor(Y[task][vid].data,requires_grad=True)
                y = Y[task][vid].requires_grad_(True)
                loss = loss_fn(O, y, C)
                param_grads = th.autograd.grad(loss, net.parameters(), create_graph=True, only_inputs=True)
                F = loss
                for g in param_grads:
                    F -= 0.5*args.lr*(g**2).sum()
                Y_grad = th.autograd.grad(F,[y], only_inputs=True)
                y = np.zeros(Y[task][vid].size(),dtype=np.float32)
                dp(y,Y_grad[0].cpu().numpy())
                Y[task][vid].requires_grad_(False)
                Y[task][vid].data = th.tensor(y,dtype=th.float).cuda() if args.use_gpu else th.tensor(y,dtype=th.float)

            # updating model parameters
            O = net(X, task)
            loss = loss_fn(O,Y[task][vid],C)
            loss.backward()
            cumloss += loss.item()
            optimizer.step()
            net.zero_grad()
    return cumloss

def eval():
    net.eval()
    lsm = nn.LogSoftmax(dim=1)
    Y_pred = {}
    Y_true = {}
    for batch in testloader:
        for sample in batch:
            vid = sample['vid']
            task = sample['task']
            X = sample['X'].cuda() if args.use_gpu else sample['X']
            O = lsm(net(X, task))
            y = np.zeros(O.size(),dtype=np.float32)
            dp(y,-O.detach().cpu().numpy())
            if task not in Y_pred:
                Y_pred[task] = {}
            Y_pred[task][vid] = y
            annot_path = os.path.join(args.annotation_path,task+'_'+vid+'.csv')
            if os.path.exists(annot_path):
                if task not in Y_true:
                    Y_true[task] = {}
                Y_true[task][vid] = read_assignment(*y.shape, annot_path)
    recalls = get_recalls(Y_true, Y_pred)
    for task,rec in recalls.items():
        print('Task {0}. Recall = {1:0.3f}'.format(task, rec))
    avg_recall = np.mean(list(recalls.values()))
    print ('Recall: {0:0.3f}'.format(avg_recall))
    net.train()

print ('Training...')
net.train()
for epoch in range(args.pretrain_epochs):
    cumloss = train_epoch(pretrain=True)
    print ('Epoch {0}. Loss={1:0.2f}'.format(epoch+1, cumloss))
for epoch in range(args.epochs):
    cumloss = train_epoch()
    print ('Epoch {0}. Loss={1:0.2f}'.format(args.pretrain_epochs+epoch+1, cumloss))

print ('Evaluating...')
eval()
