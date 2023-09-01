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
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#prepare train data
inputs=[]
for k in range(200):
    temp=torch.load(f"./assest/clean_z_25/clean_z_{k}.pt")
    inputs.append(temp)
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


# Perform binary string to one-hot encoding
codes=[]
for code in usr_list:
    encoded_text = binary_string_to_one_hot(code)
    codes.append(encoded_text)





#loss func for encoder(model A)
#outputs: output for all codes
def distance_loss(outputs):
    # 计算余弦相似度矩阵
    # cos_sim_matrix = torch.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(0), dim=2)
    cos = nn.CosineSimilarity(dim=1)
    # 计算相似度矩阵中每对输出之间的距离损失
    loss = 0.0
    for (i, j) in combinations(range(len(outputs)), 2):
        # loss -= cos_sim_matrix[i, j]  # 最大化余弦相似度，相当于最小化其负数
        # loss-=cos(outputs[i], outputs[j])
        print(loss)
        loss-=torch.norm(outputs[i] - outputs[j], p=2)
    return loss


    
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

input_size = inputs[0].shape[-1]
hidden_size = 64
code_size=codes[0].shape[-1]
inputs_train = inputs[:150]
# codes_train=codes[:150]
inputs_test = inputs[150:]
num_epochs = 5000
d_constraint = 6  # 差异限制

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

    root_path="./assest/encoded_z_25_d6_e5000"
    biased_input=[]
    for file_name in os.listdir(root_path):
        temp=torch.load(os.path.join(root_path,file_name))
        biased_input.append(temp['biased_logit'])

    biased_input=torch.stack(biased_input).to(device)
    sm=nn.Softmax()
    
    
    # 从概率分布中进行采样
    
    
    output_length = biased_input.shape[1]
    vac_size = biased_input.shape[-1]
    hidden_size = 64
    # output_size=biased_input[0].shape[-1]
    expander = Expander(output_length,hidden_size,vac_size).to(device)
    expander_optimizer = optim.SGD(expander.parameters(), lr=0.001)
    
    train_start_time=time.time()
    for epoch in range(num_epochs):
        for j in range(biased_input.shape[0]):
            biased_pd=sm(biased_input[j])
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
            # ipdb.set_trace()
            expander_loss = expander_criterion(biased_input[j], simulated_z)/(10**10)
            
            
            expander_optimizer.zero_grad()
            expander_loss.backward()
            expander_optimizer.step()
        if (epoch + 1) % 1 == 0:
            # print(simu_pd[0])
            print(f"-----Epoch [{epoch+1}/{num_epochs}], loss:{expander_loss.item()}, time used: {time.time()-train_start_time}")
    torch.save(expander.state_dict(), f"./assest/models/expander_d{d_constraint}_e{num_epochs}.pt")



# train_autoencoder()
# test_autoencoder()
train_expander()







