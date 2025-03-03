import os
import time
from data_processing import load_data, save_to_pickle, load_pickle, target_to_auxiliary
from prompting import load_model, prompt_model_sg
from translation import load_translation_model, auxiliary_to_target, translate_to_tgt_batched_ctranslate
from metrics import load_similarity_model, process_answers, calculate_metrics, NumpyEncoder, get_similarity
from visualization import load_all_results, plot_composite_accuracy_comparison, plot_coverage_accuracy_curves, \
    calculate_auc_metrics
import json
import numpy as np
from tqdm import tqdm
from config import *


def main(n_samples, seed):
    print(f"Loading the eval set (seed {seed})...")
    tgt_lang_data, aux_langs_dict = load_data(seed=seed, n_samples=n_samples)

    base_dir = f"MKA-{n_samples}"
    for _, (lang, _) in tgt_lang_data.items():
        os.makedirs(f"{base_dir}/{lang}/intermediate_files", exist_ok=True)
        os.makedirs(f"{base_dir}/{lang}/results", exist_ok=True)
        os.makedirs(f"Baseline/", exist_ok=True)

    nllb, nllb_tokenizer = load_translation_model(device)
    sentence_transformer = load_similarity_model()

    print("Running the baseline without the pipeline...")
    baseline_prompts = []
    for tgt_lang, (lang_folder, (questions, options, answers)) in tgt_lang_data.items():
        baseline_prompts.append(
            [f"Question: {p[:512]}\n\nChoices: {o[:512]}\n\nCorrect Answer: " for p, o in zip(questions, options)])

    for prompt_model in tqdm(prompt_models):
        prompt_model_name = prompt_model.split("/")[1]
        llm, sg_generate = load_model(prompt_model)
        prompt_model_accuracies = baseline_run(baseline_prompts, prompt_model_name, tgt_lang_data,
                                               sg_generate,
                                               sentence_transformer)
        llm.shutdown()
        with open(f"Baseline/results_{prompt_model_name}.json", "w") as f:
            json.dump(prompt_model_accuracies, f, cls=NumpyEncoder)
        time.sleep(3)

    print("Translating prompts to auxiliary languages...")
    for tgt_lang, (lang, (prompts, options, answers)) in tgt_lang_data.items():
        for task, aux_langs in aux_langs_dict.items():
            print(tgt_lang, " ==> ", aux_langs)

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

    for prompt_model in prompt_models:
        prompt_model_name = prompt_model.split("/")[1]
        llm, sg_generate = load_model(prompt_model)

        print(f"Prompting the model: {prompt_model_name}...")
        for tgt_lang, (lang, _) in tqdm(tgt_lang_data.items()):
            print(f"{tgt_lang}")
            for task, aux_langs in aux_langs_dict.items():
                translated_prompts = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{n_samples}_translated_prompts.pkl')
                answers, prompts_list = prompt_model_sg(translated_prompts, len(aux_langs), sg_generate,
                                                        add_instruction=True)
                save_to_pickle(answers,
                               f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_prompt_answers.pkl')
                torch.cuda.empty_cache()

        llm.shutdown()
        time.sleep(5)

        print("Translating the answers back to target languages...")
        for tgt_lang, (lang, _) in tgt_lang_data.items():
            print(f"==> {tgt_lang}")
            for task, aux_langs in aux_langs_dict.items():
                answers = load_pickle(
                    f'{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_prompt_answers.pkl')
                max_length = 384 if task == "low_res" else 256
                answer_translations = auxiliary_to_target(answers, aux_langs, tgt_lang, nllb, nllb_tokenizer,
                                                          max_length)
                save_to_pickle(answer_translations,
                               f"{base_dir}/{lang}/intermediate_files/{task}_{prompt_model_name}_{n_samples}_answer_translations.pkl")
                torch.cuda.empty_cache()

        print("Processing results and running analysis...")
        for tgt_lang, (lang, (prompts, options, true_answers)) in tqdm(tgt_lang_data.items()):
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
                    model_responses, tgt_lang, aux_langs, sentence_transformer
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
                    "metadata": {"model": prompt_model, "translation_model": "nllb-200-distilled-1.3B",
                                 "num_samples": n_samples},
                    "samples": samples,
                    "runs": runs,
                }
                with open(f"{base_dir}/{lang}/results/{task}_{prompt_model_name}_{n_samples}_{similarity_model}.json",
                          "w") as f:
                    json.dump(data, f, cls=NumpyEncoder)

    # Visualization
    models = [model.split("/")[1] for model in prompt_models]
    results = load_all_results(base_dir, models)
    plot_composite_accuracy_comparison(results)
    calculate_auc_metrics(results)
    plot_coverage_accuracy_curves(results)


def baseline_run(prompts, prompt_model_name, tgt_lang_data, sg_generate, sentence_transformer):
    def get_model_correctness(model_answers, true_answers):
        return get_similarity(model_answers, true_answers, sentence_transformer) > answer_similarity_threshold

    answers, prompts_list = prompt_model_sg(prompts, 6, sg_generate, add_instruction=True)

    save_to_pickle(answers, f"Baseline/{prompt_model_name}_responses_{n_samples}.pkl")
    # answers = load_pickle(f"Baseline/{prompt_model_name}_responses_{n_samples}.pkl")

    model_answers_language_wise = list(zip(*answers))

    options_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    accuracies = {}

    for i, (tgt_lang, (lang_folder, (prompts, options, true_answers))) in enumerate(tgt_lang_data.items()):
        model_answers = list(model_answers_language_wise[i])

        true_answers_idx = []
        for correct in true_answers:
            if tgt_lang != "eng_Latn":
                true_answers_idx.append(options_map[correct])
            else:
                true_answers_idx.append(correct)

        true_answers_strings = [options[i][idx] for i, idx in enumerate(true_answers_idx)]

        model_correctness = get_model_correctness(model_answers, true_answers_strings)
        accuracy = np.mean(model_correctness)

        accuracies[tgt_lang] = accuracy

    return accuracies


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 0:
        n_samples = int(sys.argv[1])
        seed = int(sys.argv[2])

        main(n_samples, seed)
    else:
        print("Not enough arguments provided")
