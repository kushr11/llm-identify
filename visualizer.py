import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import ipdb
z=torch.load(f"./assest/clean_z_{1}.pt").cpu()
vocab_permutation = torch.randperm(z.shape[-1])
rand_ids = vocab_permutation[z.shape[-1]//10]
newz=copy.deepcopy(z)

x= [i for i in range(z.shape[-1])]
sm=nn.Softmax(dim=1)

for i in range(z.shape[0]):
    
    newz[i][rand_ids]+=4
    pdo=sm(z.float())
    pdr=sm(newz.float())
    # print(pdo[i][2],pdr[i][2])
    pdo[i][pdo[i] != pdo[i]] = 0
    pdr[i][pdr[i] != pdr[i]] = 0  #nan ->0
    # a=torch.log(pdo[i])
    # b=torch.log(pdr[i])
    # # a[a!=a]=0
    # # b[b!=b]=0
    # print(a)
    # print(i,(a-b).mean())
    # continue
    plt.plot(x,torch.log(pdo[i]),color = 'r',alpha=0.6,label="origin")#s-:方形
    plt.plot(x,torch.log(pdr[i]),color = 'g',alpha=0.6,label="new")#o-:圆形
    plt.xlabel("ids")#横坐标名字
    plt.ylabel("pds")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.show()
    plt.savefig(f"./output/out{i}.png") #save as jpg
    plt.clf()