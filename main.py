#!/usr/bin/python3

import os,argparse,random,numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#,1'

import torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,torch.utils.data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import random_split,DataLoader
from collections import OrderedDict

from dataset import MyDataset, collate_fn
from model import MLP, resnet50x1, resnet50x2, resnet50x4

from itertools import groupby
from tqdm import tqdm

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(epoch, model, loss, accu):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'loss': loss,
        'accu': accu
    }
    torch.save(state, f'./save/{epoch}_{loss:.4f}_{accu:.4f}')

def load_dataset(batchsize):
    tdataset = MyDataset(train=True)
    vdataset = MyDataset(test=True)
    #tdataset, vdataset = random_split(dataset, [len(dataset)-80,80])
    #tdataset, vdataset = random_split(dataset, [16,16])
                                                
    tloader = DataLoader(
        tdataset,#Subset(tdataset, list(range(16))),
        batch_size=batchsize, 
        shuffle=True, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=4,
        collate_fn=collate_fn
    )
    vloader = DataLoader(
        vdataset, 
        batch_size=batchsize, 
        shuffle=True, 
        sampler=None, 
        batch_sampler=None, 
        num_workers=0,
        collate_fn=collate_fn
    )
    return tloader,vloader

def load_network(encodername,numhiden,lr,device,mycheckpoint=None):
    assert encodername in ('resnet50x1','resnet50-2x.pth','resnet50-4x.pth')

    if encodername == 'resnet50x1':
        encoder = resnet50x1().to(device)
        checkpoint = torch.load("resnet50-1x.pth")
    elif encodername == 'resnet50x2':
        encoder = resnet50x2().to(device)
        checkpoint = torch.load("resnet50-2x.pth")
    elif encodername == 'resnet50x4':
        encoder = resnet50x4().to(device)
        checkpoint = torch.load("resnet50-4x.pth")

    encoder.load_state_dict(checkpoint['state_dict'])
    encoder.to(device)
    encoder.eval()

    num_in = 512*4*int(encodername[-1])
    dsMLP = MLP(num_in, numhiden).to(device)
    
    optimizer = optim.AdamW(dsMLP.parameters(), lr=lr)

    iepoch = 0
    if not mycheckpoint is None:
        if os.path.exists(mycheckpoint):
            cp = torch.load(mycheckpoint)
            iepoch = int(cp['epoch'])+1

            #下面这一行是为了消除并行训练时，保存的网络参数名字会多出来一个 "xxx.module.xxxx"
            state_dict = OrderedDict( [(k.replace('module.',''),v) for k,v in cp['model'].items()] )
            dsMLP.load_state_dict(state_dict, strict=True)

    if len(os.environ['CUDA_VISIBLE_DEVICES'])>1:
        dsMLP = torch.nn.DataParallel(dsMLP)

    return iepoch,encoder,dsMLP,optimizer

def accuracy(prd,tgt,n):
    idx0 = (tgt==1.0).nonzero()
    idx1 = (prd>=0.85).nonzero()

    y = idx0.cpu().numpy().tolist()
    x = idx1.cpu().numpy().tolist()

    hitcount = 0
    for k,items in groupby(y,key=lambda v:v[0]):
        yval = [v[1] for v in items]
        xval = set([v[1] for v in x if v[0] == k])

        for v in yval:
            if v in xval:
                hitcount += 1
                break

    return (hitcount / n.shape[0]) * 100
   
def train(epoch,encoder,DSmlp,loader,optimizer,criterion):

    losses = AverageMeter()  # loss (per word decoded)
    accurs = AverageMeter()  # top5 accuracy
    DSmlp.train()

    if isinstance(DSmlp,nn.DataParallel):
        device = next(DSmlp.module.parameters()).device
    else:
        device = next(DSmlp.parameters()).device

    # Batches
    total = len(loader)
    tmpl = '[{0}][{1}/{2}] LOSS:{loss.val:.4f}({loss.avg:.4f}) ACCURCY:{accur.val:.3f}({accur.avg:.3f})'
    msg = tmpl.format(epoch, 0, total, loss=losses, accur=accurs)
    with tqdm( total=total,desc=msg, ascii=True ) as bar:
        for i_batch,item in enumerate(loader):
            data,target,target_lengths = item

            input = data.to(device) / 255.0
            label = target.to(device)
            
            with torch.no_grad():
                feature = encoder(input.float())

            output = DSmlp(feature)
            #loss = F.cross_entropy(output, label)
            loss = criterion(output,label)
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(DSmlp.parameters(), 5.0, 2)
            optimizer.step()
            
            accu = accuracy(output, label,target_lengths)
            accurs.update(accu)

            msg = tmpl.format(epoch, i_batch, total, loss=losses,accur=accurs)
            bar.set_description(msg)
            bar.update()
    return losses.avg

def validate(encoder,DSmlp,loader,criterion):
    losses = AverageMeter()  # loss (per word decoded)
    accurs = AverageMeter()  # top5 accuracy
    DSmlp.eval()

    if isinstance(DSmlp,nn.DataParallel):
        device = next(DSmlp.module.parameters()).device
    else:
        device = next(DSmlp.parameters()).device

    # Batches
    total = len(loader)
    tmpl = '[Evaluate][{0}/{1}] LOSS:{loss.val:.4f}({loss.avg:.4f}) ACCURCY:{accur.val:.3f}({accur.avg:.3f})'
    msg = tmpl.format(0, total, loss=losses, accur=accurs)
    with tqdm( total=total,desc=msg, ascii=True ) as bar:
        for i_batch,item in enumerate(loader):
            data,target,target_lengths = item

            input = data.to(device) / 255.0
            label = target.to(device)
            
            with torch.no_grad():
                feature = encoder(input.float())
                output = DSmlp(feature)
                loss = criterion(output,label)
                losses.update(loss.item())
            
            accu = accuracy(output, label,target_lengths)
            accurs.update(accu)

            msg = tmpl.format(i_batch, total, loss=losses,accur=accurs)
            bar.set_description(msg)
            bar.update()
    return losses.avg

def main():
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',type=str,default='resnet50x1',help='type of encoder: resnet50x1, resnet50x2 or resnet50x4')
    parser.add_argument('--epochs',type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size',type=int, default=16, help='batch size')
    parser.add_argument('--num_hid',type=int, default=1500, help='num of downstream MLP hidden nodes')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')    
    #parser.add_argument('--checkpoint',type=str,default='./save/94_15.5928_0.0000')
    #parser.add_argument('--checkpoint',type=str)
    args = parser.parse_args()

    train_loader,valid_loader = load_dataset(args.batch_size)
    iepoch,encoder,model,optimizer = load_network(args.encoder,args.num_hid,args.lr,device)#,args.checkpoint)
    
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()

    bestloss = 100000
    for epoch in range(iepoch,args.epochs):
        vloss = train(epoch,encoder,model,train_loader,optimizer,criterion)
        vaccu = validate(encoder,model,valid_loader,criterion)
        if vloss < bestloss:
            save_model(epoch,model,vloss,vaccu)
            bestloss = vloss

if __name__ == '__main__':
    main()
