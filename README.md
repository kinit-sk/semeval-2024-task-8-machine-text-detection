# KInIT at SemEval-2024 Task 8: Fine-tuned LLMs for Multilingual Machine-Generated Text Detection
Source code for replication of the detection system, ranked fourth at SemEval-2024 Task 8 Subtask A in the multilingual track.

## Cite
If you use the data, code, or the information in this repository, cite the following paper.
```bibtex
@inproceedings{spiegel-macko-2024-kinit,
    title = "{KI}n{IT} at {S}em{E}val-2024 Task 8: Fine-tuned {LLM}s for Multilingual Machine-Generated Text Detection",
    author = "Spiegel, Michal  and Macko, Dominik",
    booktitle = "Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.semeval-1.84",
    doi = "10.18653/v1/2024.semeval-1.84",
    pages = "558--564",
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
2. For the integration and usage with the official scoring scipts, clone the official [SemEval-2024 Task 8](https://github.com/mbzuai-nlp/SemEval2024-task8) repository, copy the official data to the data folder as described in the official repository, and copy the content of this repository to the subtaskA folder.
   ```
   git clone https://github.com/mbzuai-nlp/SemEval2024-task8.git
   cd SemEval2024-task8
   ```
3. Or just clone this repository and use it independently.
   ```
   git clone https://github.com/kinit-sk/semeval-2024-task-8-machine-text-detection.git
   cd semeval-2024-task-8-machine-text-detection
   ```
## Code Usage
1. To retrain the Mistral-7B model, run the following code (data needs to be downloaded as described in Step 2 of the Installation). Similarly, run the code to retrain Falcon-7B.
   ```
   python3 baseline/transformer_peft.py --train_file_path data/subtaskA_train_multilingual.jsonl --test_file_path data/subtaskA_test_multilingual.jsonl --prediction_file_path predictions/mistral_test_predictions_probs.jsonl --subtask A --model 'mistralai/Mistral-7B-v0.1'
   ```
2. To regenerate statistical metrics, use the [IMGTB framework](https://github.com/michalspiegel/IMGTB).
3. For LLM2S3 ensembling, use the provided Jupyter notebook [script](https://github.com/kinit-sk/semeval-2024-task-8-machine-text-detection/blob/main/LLM2S3.ipynb).
