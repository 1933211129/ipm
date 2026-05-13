---
configs:
  - config_name: full_v1.5_default
    data_files:
      - split: default
        path: "data/full_v1.5_default.csv"
  - config_name: full_v1
    data_files:
      - split: default
        path: "data/full_v1.csv"
---
Taken from https://huggingface.co/datasets/causal-nlp/CLadder (the original was broken as of and not usable within the [lm_eval_harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main))
# Citation
>@inproceedings{jin2023cladder,  
    author = {Zhijing Jin and Yuen Chen and Felix Leeb and Luigi Gresele and Ojasv Kamal and Zhiheng Lyu and Kevin Blin and Fernando Gonzalez and Max Kleiman-Weiner and Mrinmaya Sachan and Bernhard Sch{\"{o}}lkopf},  
    title = "{CL}adder: {A}ssessing Causal Reasoning in Language Models",  
    year = "2023",  
    booktitle = "NeurIPS",  
    url = "https://openreview.net/forum?id=e2wtjx0Yqu",  
}