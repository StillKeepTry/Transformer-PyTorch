# Introduction
This project provides a PyTorch implementation about [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) based on [fairseq-py](https://github.com/facebookresearch/fairseq-py) (An official toolkit of facebook research). You can also use official code about *Attention is all you need* from [tensor2tensor](https://github.com/tensorflow/tensor2tensor).

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
Feel grateful for the contribution of the facebook research and google research. **Besides, if you get benefits from this repository, please give me a star.**

# Details

## How to install Transformer-PyTorch
You first need to install PyTorch >= 0.4.0 and Python = 3.6. And then
```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

Generating binary data, please follow the script under [data/](data/), i have provide a [run script](run_iwslt14_transformer.sh) for iwslt14.

# Results

## IWSLT14 German-English
This dataset contains 160K training sentences. We recommend you to use `transformer_small` setting. The beam size is set as 5. The results are as follow:

|Word Type|BLEU|
|:-:|:-:|
|10K jointly-sub-word|31.06|
|25K jointly-sub-word|32.12|

Please try more checkpoint, not only the last checkpoint.

## Nist Chinese-English

This dataset contains 1.25M training sentences. We learn a 25K subword dictionary for source and target languages respectively. We adopt a `transformer_base` model setting. The results are as follow:

||MT04|MT05|MT06|MT08|MT12|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Beam=10|40.67|40.57|38.77|32.26|31.04|

## WMT14 English-German
This dataset contains 4.5M sentence pairs. Wait ...

## WMT14 English-French
This dataset include 36M sentence pairs. We learned a 40K BPE for english and french. Beam size = 5. And 8 GPUs ared used in this task. For base model setting, the batch size is 4000 for each gpu.

|Steps|BLEU|
|:-:|:-:|
|2w|34.42|
|5w|37.14|
|12w|38.72|
|17w|39.06|
|21w|39.30|

And For big model, the batch size is 3072 for each gpu. The result is as:

|Steps|BLEU|
|:-:|:-:|
|5.5w|38.00|
|11w|39.44|
|16w|40.21|
|27w|40.46|
|30w|40.76|

Limited to resource, i just conduct experiment only once on Big model setting and do not try more parameters such as learning rate. I think you can produce better performance if you have rich GPUs.

## Note
> * This project is only maintained by myself. Therefore, there still exists some minor errors in code style.
> * Instead of adam, i try NAG as the default optimizer, i find this optimized method can also produce better performance.
> * If you have more suggestions for improving this project, leaving message under issues.

Our many works are built upon this project, include:
> * Double Path Networks for Sequence to Sequence Learning, (COLING 2018)
> * Other submitted papers.

# License
fairseq is BSD-licensed. The released codes modified the original fairseq are BSD-licensed. The rest of the codes are MIT-licensed.
