

# ProGAN



[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)の再現実装。

2017年。  
ICLR 2018。  

GANで高解像度の画像生成ができるようにした。

## セットアップ

 使ったライブラリ

- pytorch: 1.13.0
- torchvision: 0.14.0
- cudatoolkit: 11.7
- pyyaml: 6.0
- tensorboard: 1.15
- tqdm: 4.64.1


onnx export時

- onnx: 12.0.0
- onnxruntime: 1.13.1


## コマンド

学習を実行

```python
python train.py
```



onnxにexport
```python
python export_onnx.py
```




?
```
python.exe -m pip install -U torch-tb-profiler
```
