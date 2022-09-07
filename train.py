import json
import csv
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import default_data_collator, Trainer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
from datasets import load_metric, load_dataset
import argparse
import sacrebleu


tokenizer = AutoTokenizer.from_pretrained('t5-small')
tokenizer.pad_token = tokenizer.eos_token


def main(args):
    # Dataset
    #print('reading dataset')
    max_input_length = args.max_input_len
    max_target_length = args.max_output_len

    def preprocess_function(examples):
        inputs = [" ".join(ex) for ex in examples[args.input_column]]
#        for ex in examples[args.input_column]:
#            print(ex)
        #print(inputs)
        targets = [ex for ex in examples[args.target_column]]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True, padding='max_length',
            add_special_tokens=True,
        )
        #print(model_inputs)
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True, padding='max_length',
                add_special_tokens=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=[args.input_column, args.target_column],
        )
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=[args.input_column, args.target_column],
        )

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
        )
    
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Train model
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        #evaluation_strategy="steps",
        #eval_steps=args.eval_steps,
        load_best_model_at_end=False,
        logging_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        metric_for_best_model="eval_loss",
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.train_bsize,
        per_device_eval_batch_size=args.eval_bsize,
        label_smoothing_factor=0.1,
        #eval_accumulation_steps=10,
        weight_decay=0.01,
        learning_rate=args.lr,  
        resume_from_checkpoint=args.resume_from_checkpoint             # strength of weight decay
    )
        # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        #eval_dataset=eval_dataset if args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #data_collator=default_data_collator,
        #compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

  
    # test
#    model.to('cpu')
#    inputs = tokenizer(test_dataset['inputs'], return_tensors="pt", padding=True)
#    output_sequences = model.generate(
#        input_ids=inputs["input_ids"],
#        attention_mask=inputs["attention_mask"],
#    )
#    predictions = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
#    predictions = [pred.strip() for pred in predictions]
#    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
#    with open(output_prediction_file, "w", encoding="utf-8") as writer:
#        writer.write("\n".join(predictions))
#    print(len(predictions), len(test_dataset["target"]))
#    bleu = sacrebleu.corpus_bleu(predictions, test_dataset["target"])
#    print(bleu.score)

    trainer.save_model(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train_file", default=None, type=str
    )
    parser.add_argument(
        "--validation_file", default=None, type=str
    )
    parser.add_argument(
        "--test_file", default=None, type=str
    )
    parser.add_argument(
        "--model_name_or_path", default='t5-small', type=str, help="model to finetune"
    )
    parser.add_argument(
        "--output_dir", default='out_t5', type=str, help="dir to save finetuned model"
    )
    parser.add_argument(
        "--max_epoch", default=50, type=int, help="total number of epoch"
    )
    parser.add_argument(
        "--input_column", default=None, type=str, help="column of input"
    )
    parser.add_argument(
        "--target_column", default=None, type=str, help="column of target"
    )
    parser.add_argument(
        "--train_bsize", default=8, type=int, help="training batch size"
    )
    parser.add_argument(
        "--eval_bsize", default=16, type=int, help="evaluation batch size"
    )
    parser.add_argument(
        "--max_train_samples", default=None, type=int, help="training batch size"
    )
    parser.add_argument(
        "--max_eval_samples", default=None, type=int, help="training batch size"
    )
    parser.add_argument(
        "--save_steps", default=5000, type=int
    )
    parser.add_argument(
        "--eval_steps", default=5000, type=int
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_predict", action="store_true"
    )
    parser.add_argument(
        "--max_input_len", default=120, type=int
    )
    parser.add_argument(
        "--max_output_len", default=30, type=int
    )
    parser.add_argument(
        "--lr", default=5e-5, type=float
    )
    parser.add_argument(
        "--resume_from_checkpoint", default=None, type=str
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
#    if not os.path.exists(args.output_dir + "/runs"):
#        os.mkdir(args.output_dir + "/runs")
    main(args)
