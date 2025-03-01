from datasets import load_dataset
import pickle
import pandas as pd
from tqdm import tqdm

def load_data(seed=1209, n_samples=1000):
    en = load_dataset("cais/mmlu", "all").shuffle(seed=seed)
    bn = load_dataset("openai/MMMLU", "BN_BD").shuffle(seed=seed)
    sw = load_dataset("openai/MMMLU", "SW_KE").shuffle(seed=seed)
    ja = load_dataset("openai/MMMLU", "JA_JP").shuffle(seed=seed)
    yo = load_dataset("openai/MMMLU", "YO_NG").shuffle(seed=seed)
    id = load_dataset("openai/MMMLU", "ID_ID").shuffle(seed=seed)

    en_data = en["test"]["question"][:n_samples], en["test"]["choices"][:n_samples], en["test"]["answer"][:n_samples]
    bn_data = bn["test"][:n_samples]["Question"], list(zip(bn["test"][:n_samples]["A"], bn["test"][:n_samples]["B"], bn["test"][:n_samples]["C"], bn["test"][:n_samples]["D"])), bn["test"][:n_samples]["Answer"]
    yo_data = yo["test"][:n_samples]["Question"], list(zip(yo["test"][:n_samples]["A"], yo["test"][:n_samples]["B"], yo["test"][:n_samples]["C"], yo["test"][:n_samples]["D"])), yo["test"][:n_samples]["Answer"]
    sw_data = sw["test"][:n_samples]["Question"], list(zip(sw["test"][:n_samples]["A"], sw["test"][:n_samples]["B"], sw["test"][:n_samples]["C"], sw["test"][:n_samples]["D"])), sw["test"][:n_samples]["Answer"]
    jp_data = ja["test"][:n_samples]["Question"], list(zip(ja["test"][:n_samples]["A"], ja["test"][:n_samples]["B"], ja["test"][:n_samples]["C"], ja["test"][:n_samples]["D"])), ja["test"][:n_samples]["Answer"]
    id_data = id["test"][:n_samples]["Question"], list(zip(id["test"][:n_samples]["A"], id["test"][:n_samples]["B"], id["test"][:n_samples]["C"], id["test"][:n_samples]["D"])), id["test"][:n_samples]["Answer"]

    aux_langs_dict = {
        "high_res": ['eng_Latn', 'deu_Latn', 'fra_Latn', 'spa_Latn', 'zho_Hans', 'por_Latn'],
        "mid_res": ['ell_Grek', 'heb_Hebr', 'hin_Deva', 'ind_Latn', 'ukr_Cyrl', 'vie_Latn'],
        "low_res": ['tel_Telu', 'npi_Deva', 'mai_Deva', 'bho_Deva', 'yor_Latn', 'zul_Latn']
    }

    tgt_lang_data = {
        "ben_Beng": ("Bengali", bn_data),
        "eng_Latn": ("English", en_data),
        "yor_Latn": ("Yoruba", yo_data),
        "swh_Latn": ("Swahili", sw_data),
        "jpn_Jpan": ("Japanese", jp_data),
        "ind_Latn": ("Indonesian", id_data)
    }

    return tgt_lang_data, aux_langs_dict

def save_to_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def target_to_auxiliary(prompts, options, answers, tgt_lang, aux_langs, max_length, translate_to_tgt_batched_ctranslate):
    translated_prompts = []
    translated_df = pd.DataFrame({'prompt': prompts, 'option': [str(opts) for opts in options], 'answer': answers})

    for aux_lang in tqdm(aux_langs):
        prompt_translations = translate_to_tgt_batched_ctranslate(
            prompts, tgt_lang, aux_lang, batch_size=128, max_length=max_length
        )
        options_translations = translate_to_tgt_batched_ctranslate(
            [str(opts) for opts in options], tgt_lang, aux_lang, batch_size=128, max_length=max_length
        )

        translated_prompts.append(
            [f"Question: {p[:512]}\n\nChoices: {o[:512]}\n\nCorrect Answer: " for p, o in zip(prompt_translations, options_translations)]
        )

        translated_df[f"prompt_{aux_lang}"] = prompt_translations
        translated_df[f"options_{aux_lang}"] = options_translations

    return translated_prompts, translated_df