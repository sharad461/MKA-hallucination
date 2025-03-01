import os
import time
import torch
from data_processing import load_data, save_to_pickle, load_pickle, target_to_auxiliary
from prompting import load_model, prompt_model_sg
from translation import load_translation_model, auxiliary_to_target
from metrics import load_similarity_model, process_answers, calculate_metrics, NumpyEncoder
from visualization import load_all_results, plot_composite_accuracy_comparison, plot_coverage_accuracy_curves, \
    calculate_auc_metrics
import json
import numpy as np
from config import *


def main():
    print("Loading the eval set...")
    tgt_lang_data, aux_langs_dict = load_data(n_samples=n_samples)

    # Create directories
    for _, (lang, _) in tgt_lang_data.items():
        os.makedirs(f"{base_dir}/{lang}/intermediate_files", exist_ok=True)
        os.makedirs(f"{base_dir}/{lang}/results", exist_ok=True)

    # Load models
    nllb, nllb_tokenizer = load_translation_model(device)
    sentence_transformer = load_similarity_model()

    # Process each model
    for prompt_model in prompt_models:
        prompt_model_name = prompt_model.split("/")[1]
        model, tokenizer, sg_generate = load_model(prompt_model)

        # Step 1: Prepare prompts in auxiliary languages
        print("Translating prompts to auxiliary languages...")
        for tgt_lang, (lang, (prompts, options, answers)) in tgt_lang_data.items():
            for task, aux_langs in aux_langs_dict.items():
                max_length = 384 if task == "low_res" else 256
                translated_prompts, translated_df = target_to_auxiliary(
                    prompts, options, answers, tgt_lang, aux_langs, max_length,
                    lambda *args, **kwargs: translate_to_tgt_batched_ctranslate(*args, nllb=nllb,
                                                                                nllb_tokenizer=nllb_tokenizer, **kwargs)
                )
                translated_df.to_json(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{n_samples}_translated_prompts.jsonl',
                    orient='records', lines=True)
                save_to_pickle(translated_prompts,
                               f'{base_dir}/{lang}/intermediate_files/{task}_{n_samples}_translated_prompts.pkl')
                torch.cuda.empty_cache()

        # Step 2: Prompt the model
        print(f"Prompting the model: {prompt_model_name}")
        for tgt_lang, (lang, _) in tgt_lang_data.items():
            for task, aux_langs in aux_langs_dict.items():
                translated_prompts = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{n_samples}_translated_prompts.pkl')
                answers, prompts_list = prompt_model_sg(translated_prompts, len(aux_langs), sg_generate,
                                                        add_instruction=True)
                save_to_pickle(answers,
                               f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_prompt_answers.pkl')
                torch.cuda.empty_cache()

        # Step 3: Translate answers back
        for tgt_lang, (lang, _) in tgt_lang_data.items():
            for task, aux_langs in aux_langs_dict.items():
                answers = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_prompt_answers.pkl')
                max_length = 384 if task == "low_res" else 256
                answer_translations = auxiliary_to_target(answers, aux_langs, nllb, nllb_tokenizer, max_length)
                save_to_pickle(answer_translations,
                               f"{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_answer_translations.pkl")
                torch.cuda.empty_cache()

        # Step 4: Process results and calculate metrics
        for tgt_lang, (lang, (prompts, options, true_answers)) in tgt_lang_data.items():
            for task, aux_langs in aux_langs_dict.items():
                model_answers_translated = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_answer_translations.pkl').values()
                translated_prompts = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{n_samples}_translated_prompts.pkl')
                model_responses = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_prompt_answers.pkl')
                prompts_list = [list(item) for item in zip(*translated_prompts)]

                confidence_scores, ground_truths, samples = process_answers(
                    model_answers_translated, prompts, options, true_answers, prompts_list,
                    model_responses, aux_langs, sentence_transformer
                )
                save_to_pickle((confidence_scores, ground_truths, samples),
                               f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_final_tuple.pkl')

                runs = [{
                    "run_id": f"{tgt_lang}_{int(time.time())}",
                    "target_language": tgt_lang,
                    "task": task,
                    "auxiliary_languages": aux_langs,
                    "confidence_cutoff": cutoff,
                    "metrics": calculate_metrics(confidence_scores, ground_truths, cutoff)
                } for cutoff in np.arange(0, 1, 0.02)]

                data = {
                    "experiment_id": "high-res-v/s-low-res",
                    "metadata": {"model": prompt_model, "translation_model": "nllb-200-distilled-1.3B",
                                 "num_samples": n_samples},
                    "samples": samples,
                    "runs": runs,
                }
                with open(f"{base_dir}/{lang}/results/{task}_{prompt_model_name}_{n_samples}_{similarity_model}.json",
                          "w") as f:
                    json.dump(data, f, cls=NumpyEncoder)

    # Step 5: Visualization
    models = [model.split("/")[1] for model in prompt_models]
    results = load_all_results(base_dir, models)
    plot_composite_accuracy_comparison(results)
    calculate_auc_metrics(results)
    plot_coverage_accuracy_curves(results)


if __name__ == "__main__":
    main()