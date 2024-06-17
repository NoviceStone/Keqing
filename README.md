# Keqing
The implementation for paper [keqing: knowledge-based question answering is a nature chain-of-thought mentor of LLM](https://arxiv.org/abs/2401.00426)

### Overview

### Environment configuration

### 0. Preliminary
- Data Download
- Finish the Freebase Setup refer to the guidance from dki-lab and start the freebase service.
```
python3 virtuoso.py start 3001 -d virtuoso_db
```

### Question decomposition
Supervised fine-tuning with LoRA
