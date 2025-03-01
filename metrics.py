import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
from config import *


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def get_similarity(batch1, batch2, sentence_transformer):
    embedding1 = sentence_transformer.encode(batch1)
    embedding2 = sentence_transformer.encode(batch2)

    similarities = cosine_similarity(embedding1, embedding2)

    return np.diag(similarities) if len(batch1) == len(batch2) else similarities[0]


def find_most_probable_answer(answers, n_gram_range=(2, 3)):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=n_gram_range)
    X = vectorizer.fit_transform(answers)

    similarities = cosine_similarity(X)

    return np.argmax(np.mean(similarities, axis=1))


def process_answers(model_answers_translated, prompts, options, true_answers, prompts_list, model_responses, tgt_lang,
                    aux_langs,
                    sentence_transformer):
    model_answers_by_prompt = list(zip(*model_answers_translated))

    confidences, ground_truths, samples = [], [], []

    options_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for i, (answers, correct_answer) in enumerate(zip(model_answers_by_prompt, true_answers)):
        true_answer = options_map[correct_answer] if tgt_lang != "eng_Latn" else correct_answer
        true_answer_string = options[i][true_answer]

        final_answers_set = list(answers).copy()
        correct_idx_in_answers = find_most_probable_answer(answers)
        selected_model_answer = final_answers_set.pop(correct_idx_in_answers)

        answer_similarity = get_similarity([selected_model_answer] * len(final_answers_set), final_answers_set,
                                           sentence_transformer)
        adjusted_sims = np.where(answer_similarity > 0.8, answer_similarity * 1.5, answer_similarity)
        confidence = min(adjusted_sims.mean(), 1)

        confidences.append(confidence)
        true_v_model = get_similarity([selected_model_answer], [true_answer_string], sentence_transformer).item()
        correct = true_v_model > answer_similarity_threshold
        ground_truths.append(correct)

        samples.append({
            "question_id": i,
            "original_question": prompts[i],
            "translated_prompts": {lang: question for lang, question in zip(aux_langs, prompts_list[i])},
            "responses": {lang: answer for lang, answer in zip(aux_langs, model_responses[i])},
            "translated_responses": {lang: answer for lang, answer in zip(aux_langs, model_answers_by_prompt[i])},
            "translated_responses_with_scores": list(zip(final_answers_set, answer_similarity)),
            "decision": {
                "final_answer": selected_model_answer,
                "true_answer": true_answer_string,
                "confidence": confidence,
                "similarity_with_truth": true_v_model,
                "correct": correct
            }
        })

    return confidences, ground_truths, samples


def calculate_metrics(confidence_scores, ground_truths, confidence_cutoff):
    n_samples = len(confidence_scores)
    abstentions = sum(1 for conf in confidence_scores if conf < confidence_cutoff)
    answered = n_samples - abstentions

    true_confidences = sum(
        1 for conf, true in zip(confidence_scores, ground_truths) if true and conf >= confidence_cutoff)
    false_confidences = sum(
        1 for conf, true in zip(confidence_scores, ground_truths) if not true and conf >= confidence_cutoff)
    incorrectly_abstained = sum(
        1 for conf, true in zip(confidence_scores, ground_truths) if true and conf < confidence_cutoff)
    correctly_abstained = sum(
        1 for conf, true in zip(confidence_scores, ground_truths) if not true and conf < confidence_cutoff)

    answered_accuracy = true_confidences / answered if answered else 0
    correctly_abstained_rate = correctly_abstained / abstentions if abstentions else 0
    composite_accuracy = (true_confidences + correctly_abstained) / n_samples
    answer_rate = answered / n_samples

    return {
        "abstention_metrics": {
            "answered": answered,
            "abstentions": abstentions,
            "answer_rate": answered / n_samples,
            "abstention_rate": abstentions / n_samples,
            "correct_confidences": true_confidences,
            "incorrect_confidences": false_confidences,
            "correctly_abstained": correctly_abstained,
            "incorrectly_abstained": incorrectly_abstained,
            "answered_accuracy": answered_accuracy,
            "correctly_abstained_rate": correctly_abstained_rate,
            "accuracy": true_confidences / n_samples,
            "effective_accuracy": composite_accuracy * answer_rate,
        },
        "mean_confidence": np.mean(confidence_scores),
        "confidence_std": np.std(confidence_scores),
        "confidence_histogram": np.histogram(confidence_scores, bins=10, range=(0, 1))[0].tolist(),
    }


def load_similarity_model():
    similarity_model = 'paraphrase-multilingual-mpnet-base-v2'
    return SentenceTransformer(similarity_model)
