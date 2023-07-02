import pickle
import numpy as np
def gen_usr_list_dense():
    """
    generate fdense-user-watermark file
    :return: usr_list_dense.pkl
    """

    magnitude=7
    usr_list=np.empty([2**magnitude], dtype = '<U16', order = 'C')
    for i in range(2**magnitude):
        s_bin=bin(i)
        s_bin=s_bin.split('b')[1]
        while len(s_bin)<7:
            s_bin='0'+s_bin
        usr_list[i]=s_bin
    # print(usr_list[19])
    pickle.dump(usr_list, open("usr_list_dense.pkl", 'wb'))

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

def read_usr_list(user_dist):
    with open(f"usr_list_{user_dist}.pkl", 'rb') as fo:
        usr_list = pickle.load(fo, encoding='bytes')
        # print(usr_list)
    # print(usr_list[19])
    return usr_list


def compute_similarity(wm,id):
    sim=0
    wm=id[0]+wm
    while len(id)<len(wm):
        id+=id
    id=id[:len(wm)]
    for i in range(len(wm)):
        if (wm[i]==id[i]):
            sim+=1
    # print(sim,len(wm))
    return sim/len(wm)

# gen_usr_list_dense()
# read_usr_list()