# molmo-app

## 環境構築
```bash
git clone https://zaku.sys.es.osaka-u.ac.jp/k.yanagida/molmo-app.git
conda create -n molmo python=3.10
conda activate molmo
pip install -r requirements.txt
```

## 実行のオプション
### --use-vllm
transformersではなく，vLLMを使用する場合に指定．vLLMを使用すると推論速度が向上するがGPUメモリの使用量は増加する．

### --launch-mode
- local: 実行したPC内でgradioを使用する場合
- network: ローカルネットワーク内でgradioを使用する場合
- share: URLを作成して，どこからでもgradioを使用する場合


## 実行
```bash
python app.py --use-vllm --launch-mode network
```
