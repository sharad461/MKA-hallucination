n_samples = 200

prompt_models = ["CohereForAI/aya-expanse-8b", "google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-2b-it", "ssg97/hdc-labs-gemma-2-27b-it-gptq-int4"]

base_dir = f"MKA-{n_samples}"

similarity_model = 'paraphrase-multilingual-mpnet-base-v2'
answer_similarity_threshold = 0.85

prompt_instruction = """You are an expert assessment system. Analyze the question below, evaluate all choices, and select the most accurate answer.
Instructions:
1. Respond ONLY with the exact text of the correct answer, no additional text needed.
2. Consider each option independently and thoroughly.
3. Base your selection strictly on the information in the question.\n\n
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

translation_model_path = "../../../../../nllb-200-distilled-1.3B-int8"
