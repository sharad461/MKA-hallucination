from tqdm import tqdm
from sglang import Engine
import torch
from config import *


def generate_text_batched(model, tokenizer, prompts, batch_size=8, max_length=768, max_new_tokens=32, temperature=0.7):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_generated_texts = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
            device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            batch_texts = [[tokenizer.decode(output, skip_special_tokens=True)] for output in outputs]
            all_generated_texts.extend(batch_texts)

    return all_generated_texts


def prompt_model_sg(prompts, chunks, sg_generate, add_instruction=False):
    instruction = prompt_instruction if add_instruction else ""
    prompts_list = [list(item) for item in zip(*prompts)]
    prompts_flatten = [instruction + item for sublist in prompts_list for item in sublist]
    # print(prompts_flatten[:3])

    answers = sg_generate(prompts_flatten)
    answers = [answer["text"].strip() for answer in answers]

    answers_chunks = list(chunk(answers, chunks))
    answers_only = []
    no_answers = []

    for answer_chunk in answers_chunks:
        answers_per_lang = []
        no_answers_curr_lang = 0
        for answer in answer_chunk:
            try:
                A = answer.split("\n\n")[0]
                answers_per_lang.append(A)
            except Exception:
                answers_per_lang.append(answer)
                no_answers_curr_lang += 1
        no_answers.append(no_answers_curr_lang)
        answers_only.append(answers_per_lang)

    return answers_only, prompts_list


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_model(prompt_model):
    # tokenizer = AutoTokenizer.from_pretrained(prompt_model)
    # model t= AutoModelForCausalLM.from_pretrained(prompt_model, device_map="auto")
    # model.cuda()

    sglang_params = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 32}
    llm = Engine(model_path=prompt_model, mem_fraction_static=0.8, dtype=torch.float16) # gemma-2-27b-gptq-int4 requires float, others can do bfloat
    sg_generate = lambda prompts: llm.generate(prompts, sglang_params)

    return llm, sg_generate
