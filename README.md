
# Mitigating Memorization of Noisy Labels by Clipping the Model Prediction

ICML 2023: 
This repository is the official implementation of [LogitClip](https://arxiv.org/abs/2212.04055). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py cifar10 --alg clip -m wrn --noise_type symmetric --noise_rate 0.5 --exp_name test --gpu 0 --temp 1.5
```


## Citation

If you find this useful in your research, please consider citing:

    @inproceedings{wei2023logitclip,
      title={Mitigating Memorization of Noisy Labels by Clipping the Model Prediction},
      author={Wei, Hongxin and Zhuang, Huiping and Xie, Renchunzi and Feng, Lei and Niu, Gang and An, Bo and Li, Yixuan},
      booktitle={International Conference on Machine Learning},
      year={2023},
      organization={PMLR}
    }
