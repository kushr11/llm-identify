import pickle
import numpy as np
import ipdb
import torch
import math
def gen_usr_list_dense():
    """
    generate fdense-user-watermark file
    :return: usr_list_dense.pkl
    """

    magnitude=18
    usr_list=np.empty([2**magnitude], dtype = '<U256', order = 'C')
    for i in range(2**magnitude):
        s_bin=bin(i)
        
        s_bin=s_bin.split('b')[1]
        while len(s_bin)<magnitude:
            s_bin='0'+s_bin
        usr_list[i]=s_bin
    # print(usr_list[19])
    # ipdb.set_trace()
    pickle.dump(usr_list, open(f"usr_list_dense_mag{magnitude}.pkl", 'wb'))

def gen_usr_list_sparse():
    """
    generate fdense-user-watermark file
    :return: usr_list_dense.pkl
    """

    magnitude=7
    usr_list=np.empty([2**(magnitude-2)], dtype = '<U16', order = 'C')
    for i in range(2**(magnitude-2)):
        sparse_i=i*4
        s_bin=bin(sparse_i)
        s_bin=s_bin.split('b')[1]
        while len(s_bin)<7:
            s_bin='0'+s_bin
        usr_list[i]=s_bin
    # print(usr_list[19])
    pickle.dump(usr_list, open("usr_list_sparse.pkl", 'wb'))
# gen_usr_list_sparse()
def gen_usr_list_continous():
    result=torch.zeros(128,12)
    for i in range(result.shape[0]): 
        rt = torch.rand(12) #0-1
        result[i]=rt
        
    torch.save(result,'./usr_list_continuous.pt')
def read_usr_list(user_dist,mag):
    with open(f"usr_list_{user_dist}_mag{mag}.pkl", 'rb') as fo:
        usr_list = pickle.load(fo, encoding='bytes')
        # print(usr_list.shape)
    # print(usr_list[19])
    return usr_list

def group_usr_list(user_dist):
    grp_num=10
    res=[]
    for i in range(grp_num):
        res.append([])
    with open(f"usr_list_{user_dist}.pkl", 'rb') as fo:
        usr_list = pickle.load(fo, encoding='bytes')
    if len(usr_list)%10!=0:
        appended_len=len(usr_list)+grp_num-(len(usr_list)%grp_num)  
    else:
        appended_len=len(usr_list)
    for i in range(appended_len):
        if i<len(usr_list):
            res[i%grp_num].append(usr_list[i])
        else:
            res[i%grp_num].append("")
    res=np.array(res)
    pickle.dump(res, open(f"usr_list_{user_dist}_grp{grp_num}.pkl", 'wb'))

def cal_attenuation(code): 
    magnitude=len(code)
    attenuation_list=[0.2,0.5,0.7,0.9]
    idx=int(code,2)/2**magnitude
    attenuation=attenuation_list[math.floor(idx*4)] 
    # print(code,attenuation,math.floor(idx*4))
    return attenuation,math.floor(idx*4)

gen_usr_list_dense()
# print(cal_attenuation('10000'))
# read_usr_list('dense',14)

# gen_usr_list_continous()