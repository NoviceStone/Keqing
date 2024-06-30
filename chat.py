import argparse
import json
import os
import re
import sys
import time
import requests

from ast import literal_eval
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from tqdm import tqdm

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.fabric.plugins import BitsandbytesPrecision

from litgpt import GPT, Config, PromptStyle, Tokenizer
from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint
from litgpt.generate.base import generate


URL = "https://api.openai.com/v1/chat/completions"

HEADERS = {
  'Content-Type': 'application/json',
  # 填写你自己的 OpenAI 账户生成的令牌/API KEY
  'Authorization': ''
}


def request_chatgpt(question, context):
    RAG_prompt = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to "
                  "answer the users question. Extract answers from the provided triples and make sure there are no "
                  "redundant answers. The output should strictly follow the List format in Python. If you don't "
                  f"know the answer, just reply an empty List []. \nContext: {context}")

    data = dict(model="gpt-3.5-turbo", temperature=0.0, messages=[
        {
            "role": "system",
            "content": RAG_prompt,
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ])
    while True:
        try:
            response = requests.post(URL, headers=HEADERS, json=data)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            time.sleep(10)
            continue
    return response.json()["choices"][0]["message"]["content"]


def read_MetaQA_KB(kb_filepath):
    e1_map = defaultdict(list)
    with open(kb_filepath) as fin:
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("|")
            e1_map[e1].append((r, e2))
            e1_map[e2].append((r + "_inv", e1))
    return e1_map


def knowledge_retrieval(seed_entities, full_kb, all_entities):
    subgraph = defaultdict(list)
    for ent in seed_entities:
        try:
            matched_ent = get_close_matches(ent, all_entities)[0]
            retrieved_facts = full_kb[matched_ent]
            subgraph[matched_ent] = retrieved_facts
        except:
            continue
    return subgraph


def candidate_reasoning(q_template, subgraph):
    answer_list = []
    for ent, facts in subgraph.items():
        question = q_template.replace("[MASK]", f"[{ent}]")
        triples = []
        rel_types = set()
        for (rel, tail_ent) in facts:
            if rel.endswith("_inv"):
                triples.append(f"({tail_ent}, {rel[:-4]}, {ent})")
                rel_types.add(rel[:-4])
            else:
                triples.append(f"({ent}, {rel}, {tail_ent})")
                rel_types.add(rel)
        context = ", ".join(triples)

        # cost saving: no need to query ChatGPT if there is only one relation
        if len(rel_types) == 1:
            inferred_ans = json.dumps([tail_ent for (rel, tail_ent) in facts])
        else:
            inferred_ans = request_chatgpt(question, context)

        # syntax check: fix words with an apostrophe
        check_rule = re.compile(r"\b[a-zA-Z]+'[a-zA-Z]+\b")
        invalid_words = re.findall(check_rule, inferred_ans)
        if invalid_words:
            for word in invalid_words:
                word_fix = word.replace("'", "\'")
                inferred_ans = inferred_ans.replace(word, word_fix)

        print(f"\n>> Sub-question: {question}")
        print(f"Retrieved facts from KB: ")
        print("\n".join(triples))
        print(f"\nAnswers inferred by ChatGPT: {inferred_ans}")
        human_ans = input(">> Do you agree with ChatGPT's answer, if not, please enter your answer: ")

        if human_ans.lower().strip() in ("", "yes", "y"):
            answer_list += literal_eval(inferred_ans)
        else:
            answer_list += literal_eval(human_ans)
    # make sure there are no empty strings and duplicate answers
    answer_list = [ans for ans in answer_list if ans]
    return list(set(answer_list))


DECOMPOSE_INSTRUCT = (
    "Parse the user input question to several subquestions: [{\"question\": subquestion, \"id\": subquestion_id, \"dep\": dependency_subquestion_id, \"seed_entity\": seed_entity or <GENERATED>-dep_id}]. "
    "The special tag \"<GENERATED>-dep_id\" refer to the generated answer of the dependency subquestion and \"dep_id\" must be in \"dep\" list. The \"dep\" field denotes the ids of the previous prerequisite "
    "subquestions which generate a new answer entity that the current subquestion relies on. Think step by step about all the subquestions needed to resolve the user's request. Parse out as few subquestions "
    "as possible while ensuring that the answer to the input question can be derived. Pay attention to the dependencies and order among subquestions. If the user input question can't be parsed, you need to reply empty JSON []."
)


def main(args):
    # ========================================>>> Load the Knowledge Base <<<=======================================
    print("Loading the knowledge base...")
    FULL_KB = read_MetaQA_KB(args.kb_filepath)
    ALL_ENT = list(FULL_KB.keys())

    # ===============================>>> Load Fine-tuned Llama2 as the Decomposer <<<===============================
    fabric = L.Fabric(devices=1, precision="bf16-true", plugins=None)

    checkpoint_dir = Path(args.llama_checkpoint_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    )
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = 1024
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    L.seed_everything(1234)
    time.sleep(3)

    # ==========================================>>> Input Your Question <<<=========================================
    print("Launching...🚀")
    print(f"Now chatting with Keqing...🤖\nTo exit, press 'Enter' on an empty prompt.\n")
    while True:
        try:
            query = input(">> Question: ")
        except KeyboardInterrupt:
            break
        if query.lower().strip() in ("", "quit", "exit"):
            break
        # ======================================>>> Question Decomposition <<<======================================
        inp_sequence = prompt_style.apply(DECOMPOSE_INSTRUCT, input=query)
        inp_encoded = tokenizer.encode(inp_sequence, device=fabric.device)
        inp_length = inp_encoded.size(0)
        out_encoded = generate(
            model,
            inp_encoded,
            max_returned_tokens=1024,
            temperature=0.0,
            eos_id=tokenizer.eos_id
        )
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        completed_seq = tokenizer.decode(out_encoded[inp_length:])
        try:
            sub_questions = json.loads(completed_seq)
        except:
            print(f"Generated sequence:\n{completed_seq}")
            sub_questions = []
            while not sub_questions:
                fabric.print("The generated sequence cannot be converted to a Dict object in Python, "
                             "please modify it to a valid sequence in JSON format.")
                completed_seq = input(">> Modified sequence: ")
                try:
                    sub_questions = json.loads(completed_seq)
                except:
                    pass
        print(f"The original question is decomposed into {len(sub_questions)} sub-questions:")
        for element in sub_questions:
            print(f"{element['id']}. {element['question']}")

        rationales = []
        # =================================>>> Iteratively solve sub-questions <<<=================================
        for element in sub_questions:
            if element["dep"][0] != -1:
                dep_id = element["dep"][0]
                exclude_ans = rationales[dep_id]["seed_entity"]
                element["seed_entity"] = rationales[dep_id]["target_entity"]

            subgraph = knowledge_retrieval(element["seed_entity"], FULL_KB, ALL_ENT)
            answer = candidate_reasoning(element["question"], subgraph)
            if element["dep"][0] != -1:
                answer = list(set(answer).difference(set(exclude_ans)))

            rationales.append({
                "id": element["id"],
                "question": element["question"],
                "seed_entity": list(subgraph.keys()),
                "target_entity": answer
            })
        print("\n>> Based on the inference results, the final answers to the original question are:")
        print(rationales[-1]["target_entity"])
        print("\n")
        print("************************************** New Round **************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define command line arguments.
    parser.add_argument(
        "--kb_filepath",
        type=str,
        default="",
        help="Path to knowledge base.",
    )
    parser.add_argument(
        "--llama_checkpoint_dir",
        type=str,
        default="",
        help="Checkpoint directory for the fine-tuned LLM that serves as the question decomposer.",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # Launch the job.
    main(args)
