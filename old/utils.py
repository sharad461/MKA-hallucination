import torch
from tqdm import tqdm

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_most_probable_answer(answers, n_gram_range=(2, 3)):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=n_gram_range)
    X = vectorizer.fit_transform(answers)

    similarities = cosine_similarity(X)

    avg_similarities = np.mean(similarities, axis=1)
    most_central_idx = np.argmax(avg_similarities)

    # return answers[most_central_idx]
    return most_central_idx


def generate_text_batched(
        model,
        tokenizer,
        prompts,
        batch_size=8,
        max_length=512,
        max_new_tokens=64,
        num_return_sequences=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    all_generated_texts = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

            if num_return_sequences > 1:
                outputs = outputs.view(len(batch_prompts), num_return_sequences, -1)
                batch_texts = [
                    [tokenizer.decode(output, skip_special_tokens=True)
                     for output in prompt_outputs]
                    for prompt_outputs in outputs
                ]
            else:
                batch_texts = [
                    [tokenizer.decode(output, skip_special_tokens=True)]
                    for output in outputs
                ]

            all_generated_texts.extend(batch_texts)

    return all_generated_texts


def translate_to_tgt_batched_nllb(
        model,
        tokenizer,
        texts,
        src_lang,
        tgt_lang,
        batch_size=32,
        max_length=128,
        num_beams=5,
        temperature=0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    all_translated_texts = []
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        inputs["forced_bos_token_id"] = forced_bos_token_id

        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=1,
                temperature=temperature
            )

            batch_translated_texts = tokenizer.batch_decode(
                translated,
                skip_special_tokens=True
            )

            all_translated_texts.extend(batch_translated_texts)

    return all_translated_texts


def target_to_auxiliary(translator, tokenizer, prompts, options, true_answers, src_lang, aux_langs, batch_size,
                        prompt_in_tgt):
    translated_prompts = []
    translated_df = pd.DataFrame({'prompt': prompts, 'option': [str(opts) for opts in options], 'answer': true_answers})

    for aux_lang in tqdm(aux_langs):
        prompt_translations = translate_to_tgt_batched_nllb(translator, tokenizer, prompts, src_lang, aux_lang,
                                                            batch_size=batch_size)
        options_translations = translate_to_tgt_batched_nllb(translator, tokenizer, [str(opts) for opts in options],
                                                             src_lang, aux_lang,
                                                             batch_size=batch_size)

        translated_prompts.append([f"Question: {p}\n\nChoices: {o}\n\nCorrect Answer: " for p, o in
                                   zip(prompt_translations, options_translations)])

        translated_df[f"prompt_{aux_lang}"] = prompt_translations
        translated_df[f"options_{aux_lang}"] = options_translations

    if prompt_in_tgt:
        translated_prompts.append(
            [f"Question: {p}\n\nChoices: {o}\n\nCorrect Answer: " for p, o in zip(prompts, options)])

    return translated_prompts, translated_df


def get_answers(model, tokenizer, translated_prompts, instruction, batch_size, add_instruction=False):
    EOS_TOKEN = tokenizer.eos_token
    for i in range(len(translated_prompts)):
        for j in range(len(translated_prompts[i])):
            translated_prompts[i][j] += EOS_TOKEN

    instr = instruction if add_instruction else ""

    prompts_list = [list(item) for item in zip(*translated_prompts)]
    prompts_flatten = [instr + item for sublist in prompts_list for item in sublist]
    print(prompts_flatten[:5])

    print("\nPrompting the model...")
    answers = generate_text_batched(model, tokenizer, prompts_flatten, batch_size=4 * batch_size, max_new_tokens=32)

    return answers, prompts_list


def extract_answers(answers):
    answers_chunks = list(chunk(answers, 6 + 1))

    answers_only = []
    for answer_chunk in answers_chunks:
        answers_per_lang = []
        for answer in answer_chunk:
            try:
                # answers_per_lang.append(answer[0].split("Answer:")[1].split("\n\n")[0].strip())
                A = answer[0].split("Answer:")[1]
                try:
                    B = A.split("\n\n")[0].strip()
                    answers_per_lang.append(B)
                except Exception as e:
                    answers_per_lang.append(A)
            except Exception as e:
                answers_per_lang.append("")
                print("No answer for: ", answer[0])
        answers_only.append(answers_per_lang)

    return answers_only

    # answers_only = [[answer.split("Answer:")[1].strip() for answer in answer_chunk] for answer_chunk in answers_chunks]


def auxiliary_to_target(translator, tokenizer, extracted_answers, tgt_lang, aux_langs, batch_size, prompt_in_tgt):
    answers_aux_lang = list(zip(*extracted_answers))
    tgt_lang_answers = answers_aux_lang.pop(-1)  # Don't translate the answers in the tgt language

    answers_aux_lang = zip(answers_aux_lang, aux_langs)
    answer_translations = []
    for p, l in tqdm(answers_aux_lang):
        answer_translations.append(
            translate_to_tgt_batched_nllb(translator, tokenizer, p, l, tgt_lang, batch_size=batch_size))

    if prompt_in_tgt:
        answer_translations.append(tgt_lang_answers)

    return answer_translations


def handle_null(obj):
    if obj is None:
        return ''
    return obj


def get_prompts_from_df(tokenizer, df, prompts, options, aux_langs):
    # Only creating this function in case the prompt structure needs to be changed
    # Make changes in the f-strings below
    translated_prompts_ = []

    EOS_TOKEN = tokenizer.eos_token

    for aux_lang in tqdm(aux_langs):
        prompt_translations = df[f"prompt_{aux_lang}"]
        options_translations = df[f"options_{aux_lang}"]
        translated_prompts_.append([f"Question: {p}\n\nChoices: {o}\n\nCorrect Answer: " for p, o in
                                    zip(prompt_translations, options_translations)])

    translated_prompts_.append([f"Question: {p}\n\nChoices: {o}\n\nCorrect Answer: " for p, o in zip(prompts, options)])

    return translated_prompts_
