n_samples = 50

prompt_models = ["openai-community/gpt2", ]  # "CohereForAI/aya-expanse-8b", "google/gemma-2-2b-it"]

base_dir = f"MKA-SG-{n_samples}"

similarity_model = 'paraphrase-multilingual-mpnet-base-v2'
answer_similarity_threshold = 0.85

prompt_instruction = "Given below is a question and possible answers. Choose the correct answer.\n\n"

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

translation_model_path = "../../../../../nllb-200-distilled-1.3B-int8"
