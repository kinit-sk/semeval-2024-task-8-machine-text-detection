# KInIT at SemEval-2024 Task 8: Fine-tuned LLMs for Multilingual Machine-Generated Text Detection
Source code for replication of the detection system, ranked fourth at SemEval-2024 Task 8 Subtask A in the multilingual track.

## Cite
If you use the data, code, or the information in this repository, cite the following paper.
```bibtex
@inproceedings{KInITSemeval2024task8,
    title={{KInIT} at {SemEval}-2024 Task 8: Fine-tuned {LLMs} for Multilingual Machine-Generated Text Detection},
    author={Michal Spiegel and Dominik Macko},
    booktitle = {Proceedings of the 18th International Workshop on Semantic Evaluation},
    series = {SemEval 2024},
    year = {2024},
    address = {Mexico, address = {Mexico City, Mexico},
    month = {June},
    pages = {},    
    doi= {},
    misc = {https://github.com/kinit-sk/semeval-2024-task-8-machine-text-detection}           
}
```

## Source Code Structure
| File | Description |
| :- | :- |
|baseline/transformer_baseline.py|the official baseline script modified to also export machine-class probabilities|
|baseline/transformer_peft.py|the script for QLoRA PEFT fine-tunning of the input LLM|
|LLM2S3.ipynb|a Jupyter Notebook for ensembling the Falcon-7B and Mistral-7B and statistical predictions|
|predictions/*|dumped predictions for easier analysis of the LLM2S3 ensemble (without retraining the models)|

## Installation
1. Clone and install the [IMGTB framework](https://github.com/michalspiegel/IMGTB), activate the conda environment.
   ```
   git clone https://github.com/michalspiegel/IMGTB.git
   cd IMGTB
   conda env create -f environment.yaml
   conda activate IMGTB
   ```
2. Clone the official [SemEval-2024 Task 8 repository](https://github.com/mbzuai-nlp/SemEval2024-task8), copy the official data to the data folder as described in the official repository.
   ```
   git clone https://github.com/mbzuai-nlp/SemEval2024-task8.git
   cd SemEval2024-task8
   ```
3. Copy the content of this repository to the subtaskA folder.


