# n_samples = 200
# base_dir = f"MKA-{n_samples}"

prompt_models = ["CohereForAI/aya-expanse-8b", "google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct",]
                 #"google/gemma-2-2b-it", "ssg97/gemma-2-27b-it-gptq-int4"]

similarity_model = 'paraphrase-multilingual-mpnet-base-v2'
answer_similarity_threshold = 0.85

# prompt_instruction = "Given below is a question, possible choices and the correct answer.\n\n"

prompt_instruction = """You are an expert assessment system. Analyze the question below, evaluate all choices, and select the most accurate answer.
Respond ONLY with the correct answer, no additional info required.\n\n"""

import torch

device = "cuda" #if torch.cuda.is_available() else "cpu"

translation_model_path = "../nllb-200-distilled-1.3B"
