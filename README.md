# Keqing
The implementation for paper [keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM](https://arxiv.org/abs/2401.00426)

### Workflow
![keqing's workflow](/assets/workflow.jpg)

### Get Started
- Environment configuration
```bash
git clone https://github.com/NoviceStone/Keqing.git
cd Keqing
conda create -n keqing python=3.11 -y
conda activate keqing
pip install 'litgpt[all]'
```

- Data download
  - The raw datasets are available at their official websites [MetaQA](https://github.com/yuyuz/MetaQA)、[WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) and [GrailQA](https://dki-lab.github.io/GrailQA).
  - We have also provided the **processed version** of [MetaQA](https://drive.google.com/file/d/1kQ5d9ASbc53BDQDoxM2_-c5J5wkaDjz5/view?usp=sharing) tailored to Keqing, you can download it and put the unzipped `data` under the folder `Keqing`.

### Question Decomposition
**Core idea:** *complex questions can be challenging to handle, while answering simple questions is a breeze.*

LLMs are inherently endowed with powerful semantic understanding capabilities, offering us a preferred tool for parsing complex questions into simpler sub-questions. 
For KBQA, one would expect that each decomposed sub-question can be easily resolved with a single-hop inference over KG, yet this often requires some alignment between the LLM and the KG.
Therefore, we resort to [instruction fine-tuning](https://arxiv.org/abs/2203.02155) to adapt the LLM to the structured knowledge in KG for better decomposition result.

Specifically, we opt for fine-tuning **Llama2-7B** using [LoRA](https://arxiv.org/abs/2106.09685) so that it is computationally affordable even with a single graphics card (like *A6000* or *RTX8000*).
To implement the fine-tuning, an easy-to-use project [LitGPT](https://github.com/Lightning-AI/litgpt) is recommened. What we need to do is to prepare the corpus for instruction fine-tuning, below is an example to illustrate the required data format.
```JSON
{
    "instruction": "Parse the user input question to several subquestions: [{'question': subquestion, 'id': subquestion_id, 'dep': dependency_subquestion_id, 'seed_entity': seed_entity or <GENERATED>-dep_id}]...",
    "input": "the actor of Kung Fu Panda also starred in which movies?",
    "output": "[{'question': 'who acted in the movie [MASK]?', 'id': 0, 'dep': [-1], 'seed_entity': ['Kung Fu Panda']}, {'question': 'what films did [MASK] act in?', 'id': 1, 'dep': [0], 'seed_entity': ['<GENERATED>-0']}]"
}
```

**Checkpoints:** we provide the [fine-tuned LoRA weights](https://drive.google.com/file/d/19Edq-ObuouZto6_w0yMEk53uEO8StYq6/view?usp=drive_link) of Llama-2-7b on MetaQA, you can download and use it directly. But first you may need to download the base model weights for [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main). Then we can merge these two weights into a complete fine-tuned model weights using [LitGPT](https://github.com/Lightning-AI/litgpt).
```bash
git clone https://github.com/Lightning-AI/litgpt.git
cd litgpt

# put the base model (llama-2-7b) weights under this directory
mkdir checkpoints

litgpt merge_lora --checkpoint_dir out/finetune/lora-llama2-7b-metaQA-allhop/final
```

