# MKA: Model Abstention

### Requirements
```shell
pip install --upgrade pip
pip install datasets ctranslate2 sentence-transformers sglang[all]>=0.4.2.post2
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]>=0.4.2.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

### Get the translation model
```shell
ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B --output_dir nllb-200-distilled-1.3B-int8
```

### Log into Huggingface for gated model access
```shell
huggingface-cli login --token $HF_TOKEN
```

### Run
```shell
bash run.sh 200 79 # n_samples, seed
```
