# Keqing
The implementation for paper [keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM](https://arxiv.org/abs/2401.00426)

### Workflow
![keqing's workflow](/assets/workflow.jpg)

### Environment configuration

### 0. Preliminary
- Data Download
- Finish the Freebase Setup refer to the guidance from dki-lab and start the freebase service.
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

### Question decomposition
**Core idea:** *complex questions can be challenging to handle, while answering simple questions is a breeze.*

LLMs are inherently endowed with powerful semantic understanding capabilities, offering us a preferred tool for parsing complex questions into simpler sub-questions. 
For KBQA, one would expect that each decomposed sub-question can be easily resolved with a single-hop inference over KG, this usually requires some alignment between the LLM and the KG.
So we resort to supervised instruction fine-tuning (SFT) to adapt the LLM to the structured knowledge in KG.

Specifically, we Llama2 use a vey project [LitGPT](https://github.com/Lightning-AI/litgpt)

```JSON
{
    "instruction": "Parse the user input question to several subquestions: [{'question': subquestion, 'id': subquestion_id, 'dep': dependency_subquestion_id, 'seed_entity': seed_entity or <GENERATED>-dep_id}]...",
    "input": "the actor of Kung Fu Panda also starred in which movies?",
    "output": "[{'question': 'who acted in the movie [MASK]?', 'id': 0, 'dep': [-1], 'seed_entity': ['Kung Fu Panda']}, {'question': 'what films did [MASK] act in?', 'id': 1, 'dep': [0], 'seed_entity': ['<GENERATED>-0']}]"
}
```
