#!/usr/bin/python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch,torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import resnet50x1, resnet50x2, resnet50x4, MLP


checkpoint_1 = torch.load("../checkpoint/resnet50-1x.pth")
#checkpoint_2 = torch.load("../checkpoint/resnet50-2x.pth")
#checkpoint_4 = torch.load("../checkpoint/resnet50-4x.pth")
resnet_1 = resnet50x1().to(device)
#resnet_2 = resnet50x2().to(device)
#resnet_4 = resnet50x4().to(device)


resnet_1.load_state_dict(checkpoint_1['state_dict'])
resnet_1.eval()
#resnet_2.eval()
#resnet_4.eval()

dsMLP = MLP(1500)
dsMLP.train()

with torch.no_grad():
    input = torch.randn(2, 3, 100, 100)
    output = resnet_1(input.to(device))
    print(output.shape)
    print(output.device)

#import pdb
#pdb.set_trace()
    
dsMLP.to(output.device)
output = dsMLP(output)

#output = resnet_2(input)
#output = resnet_4(input)
print(output.shape)

