import os
import numpy as np
import json
from tqdm import tqdm
from utils import find_most_probable_answer, auxiliary_to_target, get_answers, target_to_auxiliary, extract_answers

import torch

import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

timestamp = datetime.datetime.now().isoformat()
n_samples = 1000
confidence_cutoff = 0.75
prompt_model = "google/gemma-2-2b-it"  # "meta-llama/Llama-3.1-8B-Instruct" #"CohereForAI/aya-expanse-8b"
translation_model = "facebook/nllb-200-distilled-1.3B"
aux_langs = ['deu_Latn', 'fra_Latn', 'spa_Latn', 'zho_Hans', 'rus_Cyrl', 'por_Latn']
tgt_lang = "eng_Latn"

prompt_in_tgt = True
batch_size = 6  # Full model # batch_size = 12 # 4-bit
max_new_tokens = 32  # For prompt answers
tmp_file_suffix = f"_{n_samples}"

ds = load_dataset("cais/mmlu", "all")
ds = ds.shuffle(seed=74)

prompts = ds["test"]["question"][:n_samples]
options = ds["test"]["choices"][:n_samples]
true_answers = ds["test"]["answer"][:n_samples]

experiment_profile = {
    "prompts": prompts,
    "options": options,
    "true_answers": true_answers,
    "tgt_lang": tgt_lang,
    "aux_langs": aux_langs,
    "string_similarity": find_most_probable_answer,  # answer choosing function
    # confidence function
    "n_langs": len(aux_langs),
    "confidence_cutoff": confidence_cutoff,
    "batch_size": 12,
    "prompt_in_tgt": True,
}

aya_languages = [
    #  "ara_Arab",
    "zho_Hans",
    "zho_Hant",
    "ces_Latn",
    "nld_Latn",
    "eng_Latn",
    "fra_Latn",
    "deu_Latn",
    "ell_Grek",
    "heb_Hebr",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kor_Hang",
    #  "fas_Arab",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "spa_Latn",
    "tur_Latn",
    "ukr_Cyrl",
    "vie_Latn"
]

string_similarity = find_most_probable_answer

translation_model_name = translation_model.split("/")[1]
prompt_model_name = prompt_model.split("/")[1]

instruction = "Given below is a question, possible choices and the correct answer.\n\n"  # Llama

folder_name = f'{prompt_model_name}-{translation_model_name}-{n_samples}'

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')


def get_similarity(batch1, batch2):
    embedding1 = sentence_transformer.encode(batch1)
    embedding2 = sentence_transformer.encode(batch2)

    similarity = cosine_similarity(embedding1, embedding2)[0]
    return similarity


if __name__ == "__main__":
    print(folder_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Load the prompting model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(prompt_model)
    model = AutoModelForCausalLM.from_pretrained(prompt_model)

    # Llama models:
    # also padding_side="left"
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    model.cuda()

    # Load the translation model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # translation_model = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(translation_model, src_lang=tgt_lang)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(translation_model)
    nllb.to("cuda")

    print("Target language: ", tgt_lang)
    print("Auxiliary languages: ", aux_langs)
    print("=======================================")

    # Step 1: Translate prompts to auxiliary languages
    print("\nPreparing prompts in auxiliary languages...")
    translated_prompts, translated_df = target_to_auxiliary(nllb, nllb_tokenizer, prompts, options, true_answers,
                                                            tgt_lang, aux_langs,
                                                            batch_size, prompt_in_tgt)

    translated_df.to_json(f'{translation_model_name}{tmp_file_suffix}_translated_prompts.jsonl', orient='records',
                          lines=True)

    with open(f'{translation_model_name}{tmp_file_suffix}_translated_prompts.pkl', 'wb') as f:
        pickle.dump(translated_prompts, f)

    torch.cuda.empty_cache()

    # Step 2: Prompt the model with the translated prompts
    with open(f'{translation_model_name}{tmp_file_suffix}_translated_prompts.pkl', 'rb') as f:
        translated_prompts = pickle.load(f)

    add_instruction = False  # Depends on the model, does it need additional instructions?
    answers, prompts_list = get_answers(model, tokenizer, translated_prompts, instruction, batch_size, add_instruction)

    with open(f'{folder_name}/prompt_answers.pkl', 'wb') as f:
        pickle.dump(answers, f)

    torch.cuda.empty_cache()

    # Step 3: Extract answers from the model's responses
    with open(f'{folder_name}/prompt_answers.pkl', 'rb') as f:
        answers = pickle.load(f)

    answers_only = extract_answers(answers)

    answer_translations = auxiliary_to_target(nllb, nllb_tokenizer, answers_only, tgt_lang, aux_langs, batch_size,
                                              prompt_in_tgt)

    with open(f'{folder_name}/answer_translations.pkl', 'wb') as f:
        pickle.dump(answer_translations, f)

    torch.cuda.empty_cache()

    # Step 4: Use a confidence cutoff to filter and process the model's performance on
    # a prompt in the target language
    confidence_cutoff = 0.75
    print("\nProcessing the answers...")

    with open(f'{translation_model_name}{tmp_file_suffix}_translated_prompts.pkl', 'rb') as f:
        translated_prompts = pickle.load(f)

    with open(f'{folder_name}/answer_translations.pkl', 'rb') as f:
        answer_translations = pickle.load(f)

    prompts_list = [list(item) for item in zip(*translated_prompts)]


    def process_model_answers(answer_translations, prompts_list, answers_only):
        answer_tgt_lang = list(zip(*answer_translations))
        samples = []

        all_langs = aux_langs + [tgt_lang]
        abstentions, correct_answers, correct_and_answered, correct_and_abstained = 0, 0, 0, 0
        for i, (answers, correct) in tqdm(enumerate(zip(answer_tgt_lang, true_answers))):
            true_correct = options[i][correct]

            # TODO: Make the confidence calculation a different function (maybe, the string selection function is also inside, so)
            final_answers = list(answers).copy()
            correct_idx = string_similarity(answers)
            model_correct = final_answers.pop(correct_idx)

            answer_similarity = get_similarity([model_correct] * len(final_answers), final_answers)
            adjusted_sims = np.where(answer_similarity > 0.8, answer_similarity * 1.5, answer_similarity)
            confidence = adjusted_sims.mean()
            confidence = 1 if confidence > 1 else confidence

            abstain = True if confidence < confidence_cutoff else False
            abstentions += int(abstain)

            true_v_model = get_similarity([model_correct], [true_correct]).item()
            correct = true_v_model > 0.85
            correct_answers += int(correct)

            correct_and_abstained += int(abstain and correct)
            correct_and_answered += int(not abstain and correct)

            try:
                samples.append({
                    "question_id": i,
                    "original_question": prompts[i],
                    "translated_prompts": {lang: question for lang, question in zip(all_langs, prompts_list[i])},
                    "responses": {lang: answer for lang, answer in zip(all_langs, answers_only[i])},
                    "translated_responses": {lang: answer for lang, answer in zip(all_langs, answer_tgt_lang[i])},
                    "translated_responses_with_scores": list(zip(final_answers, answer_similarity)),
                    "decision": {
                        "abstained": abstain,
                        "confidence": confidence,
                        "true_answer": true_correct,
                        "final_answer": model_correct,
                        "similarity_with_truth": true_v_model,
                        "correct": correct
                    }
                })
            except Exception as e:
                print(e)

        return samples, abstentions, correct_answers, correct_and_answered, correct_and_abstained


    samples, abstentions, correct_answers, correct_and_answered, correct_and_abstained = process_model_answers(
        answer_translations, prompts_list, answers_only)

    print("\nFinished processing")

    runs = [
        {
            "run_id": "",
            "target_language": {
                "code": "eng_Latn"
            },
            "auxiliary_languages": aux_langs,
            "samples": samples,
            "metrics": {
                "abstentions": abstentions,
                "abstention_rate": abstentions / n_samples,
                "total_accuracy": correct_answers / n_samples,
                "answered_accuracy": correct_and_answered / (n_samples - abstentions) if n_samples - abstentions else 0,
                "abstained_accuracy": correct_and_abstained / abstentions
                #   "accuracy_with_abstention": 0.0,
                #   "accuracy_without_abstention": 0.0,
                #   # "average_agreement_score": 0.0
            }
        }
    ]

    data = {
        "experiment_id": "high-res-v/s-low-res",
        "metadata": {
            "timestamp": timestamp,
            "model": prompt_model,
            "translation_model": translation_model,
            "dataset": "mmlu",
            "num_samples": n_samples,
        },
        "runs": runs,
    }


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)  # Convert to Python float
            if isinstance(obj, np.bool_):
                return bool(obj)  # Convert to Python bool
            return json.JSONEncoder.default(self, obj)


    with open(f"{folder_name}/experiment_results{tmp_file_suffix}.json", "w") as f:
        json.dump(data, f, cls=NumpyEncoder)
