# Introduction

This project provide a PyTorch implementation about [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) based on [fairseq-py](https://github.com/facebookresearch/fairseq-py) (An official toolkit of facebook research). You can also use office code about *Attention is all you need* from [tensor2tensor](https://github.com/tensorflow/tensor2tensor).

If you use this code about cnn, please cite:
```
@inproceedings{gehring2017convs2s,
  author    = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title     = "{Convolutional Sequence to Sequence Learning}",
  booktitle = {Proc. of ICML},
  year      = 2017,
}
```
And if you use this code about transformer, please cite:
```
@inproceedings{46201,
  title = {Attention is All You Need},
  author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
  year  = {2017},
  booktitle = {Proc. of NIPS},
}
```
Feel grateful for the contribution of the facebook research and the google research. **Besides, if you get benefits from this repository, please give me a star.**

# Details

## How to install Transformer-PyTorch
You first need to install PyTorch >= 0.4.0 and Python = 3.6. And then
```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

Generating binary data, please follow the script under [data/](data/), i have provide a run script for iwslt14.

# Results

## IWSLT14 German-English
In this dataset,  this dataset contains 160K training sentences. We recommend you to use `transformer_small` setting. The beam size is set as 5. The results are as follow:

|Word Type|BLEU|
|:-:|:-:|
|10K jointly-sub-word|31.06|
|25K jointly-sub-word|32.12|

Please try more checkpoint, not only the last checkpoint.

## Nist Chinese-English

In this dataset,  this dataset contains 1.25M training sentences. We learn a 25K subword dictionary for source and target languages respectively. We adopt a `transformer_base` model setting. The results are as follow:

||MT04|MT05|MT06|MT08|MT12|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Beam=10|40.67|40.57|38.77|32.26|31.04|

## WMT14 English-German
This dataset contains 4.5M sentence pairs. Wait ...

## WMT14 English-French
For base model, we learned a 40K BPE for english and french. Beam size = 5.

|Steps|BLEU|
|:-:|:-:|
|2w|34.42|
|5w|37.14|
|12w|38.72|
|17w|39.06|
|21w|39.30|

For big model, the result is as:

|Steps|BLEU|
|:-:|:-:|
|5.5w|38.00|
|11w|39.44|
|16w|40.21|

# License
fairseq-py is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.
