import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from itertools import combinations
import numpy as np
import ipdb
import sys
import random
import time
import os
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from pandas.core.frame import DataFrame 
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("logs")#创建实例文件夹“logs”



device = "cuda" 

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--train_code_mode",
        type=str,
        default="random",
        help="randon or all, decides if to train all code ",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="cos",
        help="cos or mae ",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate ",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="learning rate ",
    )
    args = parser.parse_args()
    return args

args=parse_args()

#prepare train data

# inputs=[]
# root_path="./assest/clean_z_25"
# for filename in os.listdir(root_path):
#     temp=torch.load(f"./assest/clean_z_25/{filename}")
#     inputs.append(temp)

unit_input=torch.load(f"./assest/clean_z_200/clean_z_0.pt")
    

with open(f"usr_list_dense.pkl", 'rb') as fo:
        usr_list = pickle.load(fo, encoding='bytes')


def binary_string_to_one_hot(text):
    encoding = np.zeros((len(text)))  # Two possible values: '0' and '1'
    for i, char in enumerate(text):
        if char == '0':
            encoding[i] = 0
        elif char == '1':
            encoding[i] = 1
        else:
            raise ValueError("Input text must only contain '0' and '1' characters.")
    return encoding

def tensor_to_binary_str(tensor):
    s=""
    for t in tensor:
        if t==1:
            s+='1'
        elif t==0:
            s+='0'
    return s

# Perform binary string to one-hot encoding
# codes=[]
# for code in usr_list:
#     encoded_text = binary_string_to_one_hot(code)
#     codes.append(encoded_text)
#codes=torch.load('./usr_list_continuous.pt')

input_size = unit_input.shape[-1]
hidden_size = 64
#code_size=codes[0].shape[-1]

num_epochs = 200
d_constraint = 6  # 差异限制
output_length=unit_input.shape[0]
vac_size=input_size
sm=nn.Softmax()

def Discrete_d_Feasibility_test():
    d=5
    V=unit_input.shape[-1]
    greenlist_size=V//2
    vocab_permutation = torch.randperm(V, device=device)
    greenlist_ids = vocab_permutation[:greenlist_size]
    discretor=0.5
    discrete_depth=args.depth
    discrete_length=greenlist_size//discrete_depth
    d_masks=[]
    green_hit=0
    depth_hit=torch.zeros(discrete_depth)
    total_s=0
    for i in range(discrete_depth):
        
        d_masks.append(greenlist_ids[i*discrete_length:(i+1)*discrete_length])
    root_path=f"./assest/clean_z_{output_length}"
    ct=0
    for file_name in os.listdir(root_path): 
        ct+=1
        
        input_z=torch.load(os.path.join(root_path,file_name))
        input_z=torch.tensor(input_z,dtype=torch.float32).to(device)
        if input_z.shape[0]<output_length:
                    continue
        for inp in input_z:
            finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
            min_value=finite_data.min()
            max_value=finite_data.max()
            inp[inp==float('-inf')]=min_value
            inp[inp==float('inf')]=max_value

        input_z=input_z.reshape(output_length,-1)
        
        for i in range(len(d_masks)):
            delta=d*discretor**i
            for j in range(input_z.shape[0]):
                input_z[j][d_masks[i]]=input_z[j][d_masks[i]]+delta
        # ipdb.set_trace()
        sm=nn.Softmax(dim=1)
        pds=sm(input_z)
        
        categorical_dist = dist.Categorical(pds)
        output_s = categorical_dist.sample().int()
        length=output_s.shape[0]

        
        for s in output_s:
            if s in greenlist_ids:
                green_hit+=1
                for j in range(discrete_depth):
                    if s in d_masks[j]:
                        depth_hit[j]+=1
        total_s+=length
        print(f"[{ct}]-th sentence, green hit: [{green_hit/total_s}], depth_hit:{(depth_hit/green_hit)}")
            
        

        
        

class Complete_process(nn.Module):
    def __init__(self, input_size, code_size, hidden_size,output_length):
        super(Complete_process, self).__init__()
        self.preprocess_input=nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.preprocess_code=nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU()
        )
        self.encoder_fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*hidden_size, input_size)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(output_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size*hidden_size, code_size)
        )
        for layer in self.preprocess_input:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.preprocess_code:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.encoder_fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    def encoder_combine(self,input_embedding,code_embedding,input,d_constraint):
        concatenateds=[]
        for k in range(input_embedding.shape[0]):
       
            concatenated = torch.cat((input_embedding[k].unsqueeze(0), code_embedding.unsqueeze(0)), dim=0)
            concatenateds.append(concatenated)
        
        concatenateds=torch.stack(concatenateds)
        output = self.encoder_fc(concatenateds)
        #constraint
        output = torch.clamp(output, max=input+d_constraint)
        return output
    def forward(self, input_data, code,d_constraint):
        
        input_embedding = self.preprocess_input(input_data)
        code_embedding = self.preprocess_code(code)
        
        encoder_output=self.encoder_combine(input_embedding,code_embedding,input_data,d_constraint)
        
        sm=nn.Softmax(dim=1)
        biased_pd=sm(encoder_output)

        categorical_dist = dist.Categorical(biased_pd)
        s = categorical_dist.sample().float()
        output=self.decoder(s)
        # output[output>0]=1
        # output[output<=0]=0
        # output=torch.sigmoid(output) 
        output=F.normalize(output,dim=0)
        # ipdb.set_trace()
        return output
    
class AutoEncoder(nn.Module):
    def __init__(self, input_size, code_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.preprocess_input=nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.preprocess_code=nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nn.ReLU()
        )
        self.encoder_fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*hidden_size, input_size)
        )
        self.decoder=nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, code_size)
        )
        for layer in self.preprocess_input:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.preprocess_code:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.encoder_fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    def encoder_combine(self,input_embedding,code_embedding,input,d_constraint):
        concatenateds=[]
        for k in range(input_embedding.shape[0]):
       
            concatenated = torch.cat((input_embedding[k].unsqueeze(0), code_embedding.unsqueeze(0)), dim=0)
            concatenateds.append(concatenated)
        
        
        concatenateds=torch.stack(concatenateds)
        output = self.encoder_fc(concatenateds)
        #constraint
        output = torch.clamp(output, max=input+d_constraint)
        return output
    def forward(self, input_data, code,d_constraint):
        
        input_embedding = self.preprocess_input(input_data)
        code_embedding = self.preprocess_code(code)
        encoder_output=self.encoder_combine(input_embedding,code_embedding,input_data,d_constraint)
        
        output=self.decoder(encoder_output)
        return output

class Expander(nn.Module):
    def __init__(self, output_length,  hidden_size,vac_length):
        super(Expander, self).__init__()
        self.expand=nn.Sequential(
            nn.Linear(output_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size,output_length*vac_length),
            
        )
        
        for layer in self.expand:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        self.output_length=output_length
        self.vac_length=vac_length
    def forward(self, input_data):
        expand_output=self.expand(input_data)
        expand_output = expand_output.reshape(self.output_length, self.vac_length)
        return expand_output






def train_autoencoder():
    #150/150
    # 创建编码器实例和优化器
    input_size = inputs[0].shape[-1]
    hidden_size = 64
    code_size=codes[0].shape[-1]
    atencoder = AutoEncoder(input_size, code_size,hidden_size)

    atencoder=atencoder.to(device)
    atencoder_optimizer = optim.SGD(atencoder.parameters(), lr=0.001)

    atencoder_criterion = nn.BCEWithLogitsLoss().to(device)

    # 训练自编码器

    train_start_time=time.time()
    for epoch in range(num_epochs):
        # 前向传播
        for j in range(len(inputs_train)):
            #train encoder
            code=codes[random.randint(0, 127)]
            input=torch.tensor(inputs_train[j], dtype=torch.float32).to(device)
            code=torch.tensor(code, dtype=torch.float32).to(device)
            #handle inf
            for inp in input:
                finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                min_value=finite_data.min()
                max_value=finite_data.max()
                inp[inp==float('-inf')]=min_value
                inp[inp==float('inf')]=max_value
            input=input.reshape(25,-1)
            input_embedding =atencoder.preprocess_input(input)
            code_embedding = atencoder.preprocess_code(code)

            enc_z=atencoder.encoder_combine(input_embedding,code_embedding,input,d_constraint)
            
            output=atencoder.decoder(enc_z)
            
            # output = atencoder(input,code)
            # outputs.append(output.squeeze())
            # outputs = torch.stack(outputs)
            if epoch==num_epochs-1:
                input_embedding = atencoder.preprocess_input(input)
                code_embedding = atencoder.preprocess_code(code)
                encoder_output=atencoder.encoder_combine(input_embedding,code_embedding,input,d_constraint)
                save_file={"biased_logit":encoder_output,'code':code}
                root_path=f'./assest/encoded_z_25_d{d_constraint}_e{num_epochs}'
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                torch.save(save_file,f"{root_path}/enc_z_{j}.pt")
                # sys.exit()
                # torch.save(code,f"./assest/encoded_z_25/code{j}.pt")
            code=code.expand(output.shape[0],-1)
            # print(output,code)
            atencoder_loss = atencoder_criterion(output, code)
            # print(f"{j}-th batch, loss={atencoder_loss.item()}")
            atencoder_optimizer.zero_grad()
            atencoder_loss.backward()
            atencoder_optimizer.step()
            
            # 打印损失
        if (epoch + 1) % 1 == 0:
            print(f"-----Epoch [{epoch+1}/{num_epochs}], loss:{atencoder_loss.item()}, time used: {time.time()-train_start_time}")
    torch.save(atencoder.state_dict(), f"./assest/models/autoencoder_d{d_constraint}_e{num_epochs}.pt")
    # torch.save(decoder.state_dict(), "./assest/models/decoderB.pt")


def test_autoencoder():
    #50/50
    succ=0
    model = AutoEncoder(input_size,code_size,hidden_size)
    model.load_state_dict(torch.load(f"./assest/models/autoencoder_d{d_constraint}_e{num_epochs}.pt"))
    model.eval().to(device)
    sim_criterion = nn.BCEWithLogitsLoss().to(device)
    for i in range(50):
        # temp=torch.load(f"./assest/encoded_z_25/enc_z_{i}.pt")
        # code=temp['code'].to(device)
        # enc_z=temp['biased_logit'].to(device)
        code=codes[random.randint(0, 127)]
        input=torch.tensor(inputs_test[i], dtype=torch.float32).to(device)
        code=torch.tensor(code, dtype=torch.float32).to(device)
        #process inf for input
        for inp in input:
                finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                min_value=finite_data.min()
                max_value=finite_data.max()
                inp[inp==float('-inf')]=min_value
                inp[inp==float('inf')]=max_value
        
        #enc_z
        input=input.reshape(25,-1)
        input_embedding =model.preprocess_input(input)
        code_embedding = model.preprocess_code(code)
        enc_z=model.encoder_combine(input_embedding,code_embedding,input,d_constraint)

        output=model.decoder(enc_z)
        minscore=100000000
        bestcode=[]
        for tempcode in codes:
            expand_tempcode=torch.tensor(tempcode).expand(output.shape[0],-1).to(device)
            loss_score=sim_criterion(output,expand_tempcode)
            if loss_score<minscore:
                minscore=loss_score
                bestcode=tempcode
        if_succ=torch.all(torch.tensor(bestcode).to(device)==code)
        if(if_succ):
            succ+=1
        print(f"batch {i},  if scucc:{if_succ}, total succ:{succ}")

def train_expander():

    root_path="./assest/encoded_z_25_d6_e200"
    # biased_input=[]
    # for file_name in os.listdir(root_path):
    #     temp=torch.load(os.path.join(root_path,file_name))
    #     biased_input.append(temp['biased_logit'].cpu())

    # biased_input=torch.stack(biased_input)
    sm=nn.Softmax()
    
    
    # 从概率分布中进行采样
    
    
    # output_length = biased_input.shape[1]
    # vac_size = biased_input.shape[-1]
    # hidden_size = 64
    # output_size=biased_input[0].shape[-1]
    expander = Expander(output_length,hidden_size,vac_size).to(device)
    expander_optimizer = optim.SGD(expander.parameters(), lr=0.001)
    
    train_start_time=time.time()
    for epoch in range(num_epochs):
        for file_name in os.listdir(root_path):
            temp=torch.load(os.path.join(root_path,file_name))
            biased_input=temp['biased_logit'].to(device)
            biased_pd=sm(biased_input.to(device))
            categorical_dist = dist.Categorical(biased_pd)
            s = categorical_dist.sample().float()
            # s=torch.reshape(s,(s.shape[0],1))
            simulated_z=expander(s)
            simu_pd=sm(simulated_z)
            ##kl divergency
            # kl_loss = nn.KLDivLoss(reduction="batchmean")
            # sim_log_pd = F.log_softmax(simu_pd, dim=1)
            # expander_loss = kl_loss(sim_log_pd,biased_pd)

            #test loss
            # expander_criterion = nn.BCEWithLogitsLoss().to(device)
            # expander_loss = expander_criterion(biased_input[j], simulated_z)
            expander_criterion = nn.MSELoss()
            expander_loss = expander_criterion(biased_input, simulated_z)/(10**5)
            
            
            expander_optimizer.zero_grad()
            expander_loss.backward()
            expander_optimizer.step()
        if (epoch + 1) % 1 == 0:
            # print(simu_pd[0])
            print(f"-----Epoch [{epoch+1}/{num_epochs}], loss:{expander_loss.item()}, time used: {time.time()-train_start_time}")
    torch.save(expander.state_dict(), f"./assest/models/expander_d{d_constraint}_e{num_epochs}_mseloss.pt")

def test_expander():
    #50/50
    succ=0
    # output_length = inputs[0].shape[0]
    # vac_size = inputs[0].shape[-1]
    # hidden_size = 64
    sim_criterion = nn.BCEWithLogitsLoss().to(device)
    # output_size=biased_input[0].shape[-1]model = AutoEncoder(input_size,code_size,hidden_size)
    
    
    
    model = AutoEncoder(input_size,code_size,hidden_size)
    model.load_state_dict(torch.load(f"./assest/models/autoencoder_d{d_constraint}_e{num_epochs}.pt"))
    expander = Expander(output_length,hidden_size,vac_size).to(device)
    expander.load_state_dict(torch.load(f"./assest/models/expander_d{d_constraint}_e{num_epochs}_mseloss.pt"))
    model.eval().to(device)
    expander.eval().to(device)
    for i in range(50):
        temp=torch.load(f"./assest/clean_z_25/clean_z_{i}.pt")
        input=torch.tensor(temp,dtype=torch.float32).to(device)
        code=codes[random.randint(0, len(codes)-1)]
        # input=torch.tensor(inputs_test[i], dtype=torch.float32).to(device)
        code=torch.tensor(code, dtype=torch.float32).to(device)
        #process inf for input
        for inp in input:
                finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                min_value=finite_data.min()
                max_value=finite_data.max()
                inp[inp==float('-inf')]=min_value
                inp[inp==float('inf')]=max_value
        input=input.reshape(25,-1)

        #enc_z
        input_embedding =model.preprocess_input(input)
        code_embedding = model.preprocess_code(code)
        enc_z=model.encoder_combine(input_embedding,code_embedding,input,d_constraint)
        #sample
        sm=nn.Softmax()
        biased_pd=sm(enc_z)
        categorical_dist = dist.Categorical(biased_pd)
        sampled_s = categorical_dist.sample().float()
        #simu z
        simulated_z=expander(sampled_s)
        # code
        output=model.decoder(simulated_z)
        #calculate similarity score
        minscore=100000000
        bestcode=[]
        loss_score_list=[]
        output_code=[]
        mapped_loss=0
        for tempcode in codes:
            expand_tempcode=torch.tensor(tempcode).expand(output.shape[0],-1).to(device)
            loss_score=sim_criterion(output,expand_tempcode)
            loss_score_list.append(loss_score)
            output_code.append(tempcode)
            if loss_score<minscore:
                minscore=loss_score
                bestcode=tempcode
        loss_score_list=torch.stack(loss_score_list)
        top_code=[]
        top_score=[]
        for j in range(10):
            index=loss_score_list.argmin()
            top_code.append(tensor_to_binary_str(output_code[index]))
            top_score.append(loss_score_list[index])
            loss_score_list[index]=10000000000
        # if_succ=torch.all(torch.tensor(bestcode).to(device)==code)
        if_top_succ=0
        # for temp in top_code:
            # if(torch.all(torch.tensor(temp).to(device)==code)):
        if tensor_to_binary_str(code) in top_code:
            if_top_succ+=1
        if(if_top_succ):
            succ+=1
        result_dic={"loss_score":torch.tensor(top_score).cpu(), "id":top_code}
        pd.get_option('display.width')
        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', None)
        # print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3}, time used: {time.time() - start_time}")
        # print(f"gen id {userid}, mapped sim {mapped_sim}")
        print(f"batch {i},  {code}, {bestcode} if scucc:{if_top_succ}, total succ:{succ}")
        # print(top_code)
        # print(top_score)
        print(DataFrame(result_dic).T)

def train_process():
    train_set=1000
    num_epochs=600
    if args.train_code_mode=='all':
        num_epochs=int(num_epochs/100)
    root_path=f"./assest/clean_z_{output_length}"
    # process_criterion = nn.BCEWithLogitsLoss().to(device)
    
    process = Complete_process(input_size,code_size,hidden_size,output_length).to(device)
    # process_optimizer = optim.SGD(process.parameters(), lr=args.lr)
    process_optimizer = optim.Adam(process.parameters(),lr=args.lr,weight_decay=0)
    scheduler=optim.lr_scheduler.StepLR(process_optimizer,step_size=1000,gamma=0.5)
    
    train_start_time=time.time()
    for epoch in range(num_epochs):
        train_idx=-1
        
        for file_name in os.listdir(root_path):
            train_idx+=1
            if(train_idx>train_set):
                break
            if args.train_code_mode=="all":
                input=torch.load(os.path.join(root_path,file_name))
                input=torch.tensor(input,dtype=torch.float32).to(device)
                if input.shape[0]<output_length:
                    continue
                for code in codes:
                    code=torch.tensor(code, dtype=torch.float32).to(device)
                    
                    for inp in input:
                        finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                        min_value=finite_data.min()
                        max_value=finite_data.max()
                        inp[inp==float('-inf')]=min_value
                        inp[inp==float('inf')]=max_value
                    input=input.reshape(output_length,-1)
                    output=process(input,code,d_constraint)
                    
                    if args.loss=='cos':
                        #cos
                        out_normalized = F.normalize(output, dim=0)
                        code_normalized = F.normalize(code, dim=0)
                        process_loss=1-F.cosine_similarity(out_normalized, code_normalized, dim=0)
                    elif args.loss=='mae':
                        #mae
                        process_loss = F.l1_loss(output, code)
                    
                    
                    process_optimizer.zero_grad()
                    process_loss.backward()
                    process_optimizer.step()
                    scheduler.step()

            else:
                code=codes[random.randint(0, 127)]
                code=torch.tensor(code, dtype=torch.float32).to(device)
                input=torch.load(os.path.join(root_path,file_name))
                input=torch.tensor(input,dtype=torch.float32).to(device)
                if input.shape[0]<output_length:
                    continue
                
                for inp in input:
                    finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                    min_value=finite_data.min()
                    max_value=finite_data.max()
                    inp[inp==float('-inf')]=min_value
                    inp[inp==float('inf')]=max_value
                input=input.reshape(output_length,-1)
                output=process(input,code,d_constraint)
                
                if args.loss=='cos':
                    #cos
                    out_normalized = F.normalize(output, dim=0)
                    code_normalized = F.normalize(code, dim=0)
                    # print(F.cosine_similarity(out_normalized, code_normalized, dim=0))
                    process_loss=1-F.cosine_similarity(out_normalized, code_normalized, dim=0)
                elif args.loss=='mae':
                    #mae
                    process_loss = F.l1_loss(output, code)
                
                
                writer.add_scalar(f"process loss_{args.loss}",process_loss,epoch)
                process_optimizer.zero_grad()
                process_loss.backward()
                process_optimizer.step()
                scheduler.step()
        if (epoch + 1) % 1 == 0:
            # print(simu_pd[0])
            print(f"-----Epoch  [{epoch+1}/{num_epochs}],lr={scheduler.get_last_lr()},lf:{args.loss} loss:{process_loss.item()}, time used: {time.time()-train_start_time}")
    torch.save(process.state_dict(), f"./assest/models/process_cont_c{args.train_code_mode}_d{d_constraint}_e{num_epochs}_l{args.loss}_lr{args.lr}.pt")     
    writer.close()

def test_process():
    succ=0
    num_epochs=350
    sim_criterion = nn.BCEWithLogitsLoss().to(device)
    model = Complete_process(input_size,code_size,hidden_size,output_length).to(device)
    model.load_state_dict(torch.load(f"./assest/models/process_cont_crandom_d6_e800_lmae_lr0.01.pt"))
    # expander = Expander(output_length,hidden_size,vac_size).to(device)
    # expander.load_state_dict(torch.load(f"./assest/models/expander_d{d_constraint}_e{num_epochs}_mseloss.pt"))
    model.eval().to(device)
    # expander.eval().to(device)
    for i in range(50):
        temp=torch.load(f"./assest/clean_z_25/clean_z_{i}.pt")
        input=torch.tensor(temp,dtype=torch.float32).to(device)
        code=codes[random.randint(0, len(codes)-1)]
        # input=torch.tensor(inputs_test[i], dtype=torch.float32).to(device)
        code=torch.tensor(code, dtype=torch.float32).to(device)
        #process inf for input
        for inp in input:
                finite_data = inp[(inp != float('inf')) & (inp != float('-inf'))]
                min_value=finite_data.min()
                max_value=finite_data.max()
                inp[inp==float('-inf')]=min_value
                inp[inp==float('inf')]=max_value
        input=input.reshape(25,-1)
        # input=input.reshape(25,-1)
        output=model(input,code,d_constraint)

        # if_succ=tensor_to_binary_str(output)==tensor_to_binary_str(code)
        loss_score_list=[]
        output_code=[]
        for tempcode in codes:
            expand_tempcode=torch.tensor(tempcode).expand(output.shape[0],-1).to(device)
            loss_score=sim_criterion(output,expand_tempcode)
            loss_score_list.append(loss_score)
            output_code.append(tempcode)
            if loss_score<minscore:
                minscore=loss_score
                bestcode=tempcode
        loss_score_list=torch.stack(loss_score_list)
        top_code=[]
        top_score=[]
        for j in range(10):
            index=loss_score_list.argmin()
            top_code.append(tensor_to_binary_str(output_code[index]))
            top_score.append(loss_score_list[index])
            loss_score_list[index]=10000000000
        # # if_succ=torch.all(torch.tensor(bestcode).to(device)==code)
        # if_top_succ=0
        # # for temp in top_code:
        #     # if(torch.all(torch.tensor(temp).to(device)==code)):
        # if tensor_to_binary_str(code) in top_code:
        #     if_top_succ+=1
        if(if_succ):
            succ+=1
        # result_dic={"loss_score":torch.tensor(top_score).cpu(), "id":top_code}
        # pd.get_option('display.width')
        # pd.set_option('display.width', 500)
        # pd.set_option('display.max_columns', None)
        # print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3}, time used: {time.time() - start_time}")
        # print(f"gen id {userid}, mapped sim {mapped_sim}")
        print(f"batch {i},  {code}, {output} if scucc:{if_succ}, total succ:{succ}")
        # print(top_code)
        # print(top_score)
        # print(DataFrame(result_dic).T)








# train_autoencoder()
# test_autoencoder()
# test_expander()
# train_expander()
# test_process()
# train_process()
Discrete_d_Feasibility_test()





