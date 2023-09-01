# User guide for llm identification

---



### Demo Usage

Firstly setup the environment : python 3.9 is recommended

```sh
pip install -r requirments.txt
```

To generate userlist, please run `gen_usr_list_dense()` or `gen_usr_list_sparse()` in utils.py, which generates user_list_dense(sparse).pkl before run `test.py` depends on your need


To generate and detect watermark:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b --user_dist dense --wm_mode combination --max_new_tokens 200 --delta 2

#To run opt-13b model, at least 2 CUDA devices should be availiable
CUDA_VISIBLE_DEVICES=0,1 python test.py --model_name_or_path facebook/opt-13b --user_dist dense --wm_mode combination --max_new_tokens 200 --delta 2

```

To evaluate perplexity:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b --user_dist dense --wm_mode combination --max_new_tokens 200 --delta 2 --ppl 1
```

To run train.py:

prepare clean logits in './assest/clean_z_25', where 25 means output length.
Then call train_autoencoder() in train.py, which produce autoencoder_dx_ex.pt in ./assest/models, dx is delta constraint, ex is train epoch, and biased logits in dir ./assest/ncoded_z_25_dx_ex .

Then call train_expander(), which takes files in ./assest/ncoded_z_25_dx_ex as input and produce expander_dx_ex.pt ./assest/models.



note: the above settings depends on RTX3090, if you use other smaller devices, please modify `load_model()` in `test.py`.





