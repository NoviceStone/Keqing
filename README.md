# Keqing
The implementation for paper [keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM](https://arxiv.org/abs/2401.00426)

### Workflow
![这是工作流程图](/assets/workflow.pdf "Keqing")

### Environment configuration

### 0. Preliminary
- Data Download
- Finish the Freebase Setup refer to the guidance from dki-lab and start the freebase service.
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

### Question decomposition
**Core idea:** *complex questions can be challenging to handle, while answering simple questions is a breeze.*

LLMs are inherently endowed with powerful semantic understanding capabilities, offering a preferred tool for parsing complex questions into simpler sub-questions. To ensure each sub-question can be resolved within a one-step inference over KB, we employ instruction fine-tuning

[LitGPT](https://github.com/Lightning-AI/litgpt)
Supervised fine-tuning with LoRA
```JSON
{
    "instruction": "Parse the user input question to several subquestions: [{'question': subquestion, 'id': subquestion_id, 'dep': dependency_subquestion_id, 'seed_entity': seed_entity or <GENERATED>-dep_id}]...",
    "input": "the actor of Kung Fu Panda also starred in which movies?",
    "output": "[{'question': 'who acted in the movie [MASK]?', 'id': 0, 'dep': [-1], 'seed_entity': ['Kung Fu Panda']}, {'question': 'what films did [MASK] act in?', 'id': 1, 'dep': [0], 'seed_entity': ['<GENERATED>-0']}]"
}
```
