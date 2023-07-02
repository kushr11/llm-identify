# User guide for llm identification

---



### Demo Usage

Firstly setup the environment : python 3.9 is recommended

```sh
pip install -r requirments.txt
```

To generate userlist, please run `gen_usr_list_dense()` or `gen_usr_list_sparse()` in utils.py, before run `test.py` depends on your need


To generate and detect watermark:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b --user_dist dense --wm_mode combination --identify_mode single --max_new_tokens 200 --delta 2
```

To evaluate perplexity:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b --user_dist dense --wm_mode combination --identify_mode single --max_new_tokens 200 --delta 2 --ppl 1
```
note: the above settings depends on RTX3090, if you use other smaller devices, please modify `load_model()` in `test.py`.


### Arguments
delta：logit的增加量，delta=0时没有水印效果

samping_temp：softmax公式的温度设置

user_distribution:  
    sparse时用户总量=32，即dense/4
    dense时用户总量为128

identify_mode:
    single：confidence最大的用户=实际生成用户表示实验成功；
    group：实际生成的用户属于confidence最大的三个用户表示实验成功

watermark mode(wm_mode):

    previous1: 计算watermark bit的时候只用该token的前一位token  **preferance=watermark的第 x位，x=[token[t-1] mod 7]**

    combination: 计算watermark bit的时候用该token的前两位token  **preferance=watermark的第 x位，x=[token[t-1]*token[t-2] mod 7] ，采用这种设置是因为有的token出现的频率高，会导致错误的用户用该token偶然算出正确的水印的频率也增加**


