from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed, BitsAndBytesConfig
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import datetime
import bitsandbytes as bnb
from peft import LoraConfig, PeftConfig, PeftModel, AutoPeftModelForCausalLM, TaskType, AutoPeftModelForSequenceClassification, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import torch
import torch.nn.functional as F

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits[:,1], labels.to(torch.float32))#, pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    floatorbfloat = torch.float16
    if 'lama' in model:
        floatorbfloat = torch.bfloat16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=floatorbfloat,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_name = model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model.config.use_cache = False
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    
    target_modules=[]
    if 'falcon' in model_name:
      target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif 'mistral' in model_name:
      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    else:
      target_modules=find_all_linear_names(model)
    
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        #task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["score"]
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    output_dir = checkpoints_path
    per_device_train_batch_size = 16 #4
    gradient_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    save_steps = 1000 #10
    logging_steps = 1000 #10
    learning_rate = 2e-5 #2e-4
    max_grad_norm = 0.3
    max_steps = 10 #500
    num_train_epochs=3 #added
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    fp16 = True
    bf16 = False
    
    if 'lama' in model_name:
        fp16 = False
        bf16 = True
    
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        #max_steps=max_steps, #for testing
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
    )
    
    max_seq_length = 512

    trainer = CustomTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        #peft_config=peft_config,
        #dataset_text_field="text",
        #max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    trainer.save_model(best_model_path)
    trainer.model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    torch.save(trainer.model.score.state_dict(), f'{best_model_path}/score-params.pt')
    
    return #skip merging
    print('Merging model...')
    model_temp = AutoPeftModelForSequenceClassification.from_pretrained(
    #model_temp = AutoPeftModelForCausalLM.from_pretrained(
        best_model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model_temp = model_temp.merge_and_unload()        
    model_temp.save_pretrained(
       best_model_path, safe_serialization=True, max_shard_size="2GB"
    )


def test(test_df, model_path, id2label, label2id):
    print('Loading model for predictions...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, trust_remote_code=True, num_labels=len(label2id), id2label=id2label, label2id=label2id, torch_dtype=torch.float16
    )
    
    #DM added
    if tokenizer.pad_token is None:
      if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
      else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    try:
      model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except:
      print("Warning: Exception occured while setting pad_token_id")

            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds, prob_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)

    args = parser.parse_args()

    random_seed = 0
    train_path =  args.train_file_path # For example 'subtaskA_train_multilingual.jsonl'
    test_path =  args.test_file_path # For example 'subtaskA_test_multilingual.jsonl'
    model =  args.model # For example 'xlm-roberta-base'
    subtask =  args.subtask # For example 'A'
    prediction_path = args.prediction_file_path # For example subtaskB_predictions.jsonl

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    ct = datetime.datetime.now()
    
    # train detector model
    fine_tune(train_df, valid_df, f"models/{model}/subtask{subtask}_{ct}/{random_seed}", id2label, label2id, model)

    # test detector model
    results, predictions, probs = test(test_df, f"models/{model}/subtask{subtask}_{ct}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions, 'probs': probs[:,1]})
    predictions_df.to_json(prediction_path, lines=True, orient='records')