# MKA: Leveraging Cross-Lingual Consensus for Model Abstention

This repository provides the code for the work done in [MKA: Leveraging Cross-Lingual Consensus for Model Abstention](https://arxiv.org/abs/2503.23687). 

The results reported in the paper can be accessed [here](https://drive.google.com/file/d/1FMka2vBwsTy-Hv3E80RE0EdKqUVdoMcC/view?usp=sharing). For `Gemma 9B`'s performance on `English` with the `low-resource` auxiliary languages set, open `English/results/low_res_gemma-2-9b-it_200_paraphrase-multilingual-mpnet-base-v2.json`. The `samples` object contains the model prompts and responses and translations and similarity scores and the MKA decision. The `runs` object has stats related to the experiments. For confidence cutoff of `0.64` for the above model-language combination, the stats are:

```
{
  "metrics": {
    "abstention_metrics": {
      "answered": 144,
      "abstentions": 56,
      "answer_rate": 0.72,
      "abstention_rate": 0.28,
      "correct_confidences": 72,
      "incorrect_confidences": 72,
      "correctly_abstained": 49,
      "incorrectly_abstained": 7,
      "answered_accuracy": 0.5,
      "correctly_abstained_rate": 0.875,
      "accuracy": 0.36,
      "effective_accuracy": 0.4356
    },
    "mean_confidence": 0.7677821376174688,
    "confidence_std": 0.2160755827962758,
  }
}
```

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
This work is part of the meta-study for the [AI Researcher Project](https://arxiv.org/abs/2409.04109) at Stanford NLP Group and was supported by them.

March 2025

