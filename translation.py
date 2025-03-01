import torch
from tqdm import tqdm
import ctranslate2
from transformers import AutoTokenizer


def translate_to_tgt_batched_ctranslate(source, src_lang, tgt_lang, nllb, nllb_tokenizer, batch_size=32,
                                        max_length=256):
    all_translated_texts = []
    nllb_tokenizer.src_lang = src_lang
    target_prefix = [tgt_lang]

    texts = [nllb_tokenizer.convert_ids_to_tokens(nllb_tokenizer.encode(sent)) for sent in source]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            results = nllb.translate_batch(batch_texts, target_prefix=[target_prefix] * len(batch_texts),
                                           max_decoding_length=max_length)
            targets = [result.hypotheses[0][1:] for result in results]
            outputs = [nllb_tokenizer.decode(nllb_tokenizer.convert_tokens_to_ids(target_)) for target_ in targets]
            all_translated_texts.extend(outputs)

    return all_translated_texts


def auxiliary_to_target(extracted_answers, aux_langs, nllb, nllb_tokenizer, max_length):
    answers_aux_lang = list(zip(*extracted_answers))
    answer_translations = {}

    for p, l in tqdm(zip(answers_aux_lang, aux_langs)):
        answer_translations[l] = translate_to_tgt_batched_ctranslate(
            p, l, "eng_Latn", nllb, nllb_tokenizer, batch_size=128, max_length=max_length
        )

    return answer_translations


def load_translation_model(device="cpu"):
    translation_model = translation_model_path
    nllb = ctranslate2.Translator(translation_model, device)
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    return nllb, nllb_tokenizer
