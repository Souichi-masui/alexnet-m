# alexnet-m
このプログラムは，Alexnetの性能を確認するために作ったプログラムです。
また，オリジナルのAlexnetを，Batch NormalizationやLeakyReLUに一部改良したモデルも記載して置きます。

## 動作環境
### <言語>
Python 3.7

### <使用フレームワーク・ライブラリ>
pytorch  
numpy  
time  
matplotlib  
pandas

## 仕様
### 学習
学習には，CIFAR10データセットを利用しています。（pytorchに組み込まれている機能を使用）
training.pyの内部で，オリジナルか修正版のAlexnetを選択することで切り替えることができる。あとは，training.pyを実行するだけで学習がスタートし，すべての学習が終えたら重みパラメータを保存したファイルをsaveフォルダに作成する。

### 評価
validation.pyを実行して，学習した際に作成した重みパラメータを用いて，評価を実施する。

