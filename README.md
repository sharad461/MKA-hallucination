# MKA: Leveraging Cross-Lingual Consensus for Model Abstention

This repository provides the code for our paper [MKA: Leveraging Cross-Lingual Consensus for Model Abstention](https://arxiv.org/abs/2503.23687).

### Requirements
```shell
pip install --upgrade pip
pip install datasets ctranslate2 sentence-transformers sglang[all]>=0.4.2.post2
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]>=0.4.2.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

### Get the translation model
Quantize if necessary using `--quantization int8`
```shell
ct2-transformers-converter --model facebook/nllb-200-distilled-1.3B --output_dir nllb-200-distilled-1.3B
```

### Log into Huggingface for gated model access
```shell
huggingface-cli login --token $HF_TOKEN
```

### Run
Supply the number of samples and the seed to use.
```shell
bash run.sh 200 97 # n_samples, seed
```

### Custom Setup
Make changes to the `config.py` file to change the prompting models,
similarity model, answer similarity threshold and the prompt to use.

### Citation
If you find this work useful, please cite:
```bib
@misc{duwal2025mka,
      title={MKA: Leveraging Cross-Lingual Consensus for Model Abstention}, 
      author={Sharad Duwal},
      year={2025},
      eprint={2503.23687},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23687}, 
}
```

### Acknowledgement
This work is part of the meta-study for the [AI Researcher Project](https://arxiv.org/abs/2409.04109) at Stanford NLP Group and was supported by them. We also thank [Zhaofeng Wu](https://zhaofengwu.github.io/), who originally contributed the idea.

March 2025

