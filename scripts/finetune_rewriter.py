import torch
import pandas as pd
import argparse
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as target_modules_map
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TrainingArguments, Seq2SeqTrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer, Seq2SeqTrainer, EarlyStoppingCallback, BitsAndBytesConfig

# Define functions

def balance_neogate(neogate_df:pd.DataFrame) -> pd.DataFrame:
    """
    Each entry in Neo-GATE has one masculine and one feminine translation. 
    We only want one gendered sentence for each row, but we want the final result to be balanced by gender. 
    This function returns a version of Neo-GATE with one gendered sentence for each row, where half the sentences are masculine and half are feminine. 

    Args:
        neogate_df: Neo-GATE in pandas format. Files obtained following https://huggingface.co/datasets/FBK-MT/Neo-GATE#adaptation. 

    Returns:
        new_df: A new balanced version of the dataset in pandas format. 
    """
    drop_m = "REF-F"
    drop_f = "REF-M"

    splitpoint = len(neogate_df) // 2

    m = neogate_df[:splitpoint].drop(columns=drop_m).rename(columns={"REF-M": "REF"})
    f = neogate_df[splitpoint:].drop(columns=drop_f).rename(columns={"REF-F": "REF"})

    m["GENDER"] = "M"
    f["GENDER"] = "F"

    new_df = pd.concat([m, f])
    
    assert len(new_df) == len(neogate_df), \
        f"Warning! The original dataset has {len(neogate_df)} rows, while the resulting one has {len(new_df)}"

    return new_df

def tokenize_simple(refs:str, tokenizer):
    """
    This function is meant for decoder-only models.
    It applies the model's tokenizer to the unified input sequence or prompt (obtained by calling add_template() on the training dataset).
    It should be applied by calling dataset.map(tokenize_simple, input_columns=[refs]). 

    Args:
        refs: The name of the dataset column containing the input sequences.
        tokenizer: The tokenizer loaded with load_model(). 

    Returns: 
        tokenized_data: Tokenizer output.
    """

    tokenized_data = tokenizer(
        text=refs,
        padding=False,
        truncation=False
    )
    return tokenized_data

def tokenize_both(refs:str, targets:str, tokenizer):
    """
    This function is meant for encoder-decoder models. 
    It applies the model's tokenizer to the both the inputs and the labels. 
    It should be applied by calling dataset.map(tokenize_both, input_columns=[refs, targets])

    Args:
        refs: The dataset column containing the input sequences.
        targets: The dataset column containing the target sequences or labels. 
        tokenizer: The tokenizer loaded with load_model(). 

    Returns: 
        tokenized_data: Tokenizer output for both input and target sequences.
    """
    tokenized_data = tokenizer(
        text=refs,
        text_target=targets,
        padding=False,
        truncation=False
    )
    return tokenized_data

def add_prefix(ds:Dataset, ref_col:str, language:str) -> Dataset:
    """
    This function adds a task-specific prefix (Rewriting task) to input sentences. 
    It is meant for encoder-decoder models, more specifically for the T5 family. 

    Args:
        ds: A single split of a Hugging Face dataset. 
        ref_col: The name of the dataset column containing the reference (input) sentences. 
    
    Returns:
        ds: A new version of the dataset where each sentence is preceded by a prefix "Riformula" or "Riscrivi" according to the language. 
    """
    new_refs = []

    for s in ds[f"{ref_col}"]:
        if language == "it":
            sent = "Riformula: {}".format(s.strip()) # Preprend task-specific prefix to inputs
        elif language == "en":
            sent = "Rewrite: {}".format(s.strip())
        new_refs.append(sent)
    
    new_ds = ds.remove_columns(f"{ref_col}").add_column(f"{ref_col}", new_refs)
    return new_ds

def add_sentinel(ds:Dataset, ref_col:str, target_col:str) -> Dataset:
    """
    This function adds a "sentinel token" (extra token ID) to reference and target sequences.
    It is meant for encoder-decoder models trained on a denoising objective: see Raffel et al. (2020), Lee et al. (2024).

    Args:
        ds (datasets.Dataset): The HuggingFace dataset.
        ref_col (str): The name of the dataset column containing input sentences.
        target_col (str): The name of the dataset column containing target sentences (labels). 

    Returns:
        Dataset: A new version of the dataset where a sentinel token '<extra_id_0>' is appended to each reference sentence (input) and prepended to each target sentence (label). 
    """
    new_refs = []
    new_targets = []

    for s in ds[f"{ref_col}"]:
        ref = f"{s.strip()} <extra_id_0>" # Append to input sentence
        new_refs.append(ref)
    for s in ds[f"{target_col}"]:
        tgt = f"<extra_id_0> {s.strip()}" # Prepend to target sentence
        new_targets.append(tgt)

    ds = ds.remove_columns(f"{ref_col}").add_column(f"{ref_col}", new_refs)
    ds = ds.remove_columns(f"{target_col}").add_column(f"{target_col}", new_targets)

    return ds

def add_template(ds:Dataset, ref_col:str, target_col:str, language:str) -> Dataset:
    """
    Create input-target pairs with a template based on [Zhang et al., 2023](https://aclanthology.org/2023.wmt-1.43/). 
    It is meant for standard decoder-only LLMs.
    The template for English looks like this: "Original sentence: <input> Rewritten sentence: <target>EOS".

    Args: 
        ds (datsets.Dataset): the dataset containing the sentences
        ref_col (str): the name of the column containing reference (input) sentences
        target_col (str): the name of the column containing target sentences (labels)
    
    Returns:
        Dataset: A new version of the dataset with a single column containing both input and target sentences. 
    """
    refs = ds[f"{ref_col}"]
    targets = ds[f"{target_col}"]
    
    templates = []
    for pair in list(zip(refs, targets)):
        ref = pair[0]
        tgt = pair[1]

        if language == "it":
            template = f"Frase originale: <{ref.strip()}> Riformulazione: <{tgt.strip()}></s>"
        elif language == "en":
            template = f"Original sentence: <{ref.strip()}> Rewritten sentence: <{tgt.strip()}></s>"
        templates.append(template)

    new_ds = ds.remove_columns([f"{ref_col}", f"{target_col}"]).add_column(f"{ref_col}", templates)
    
    return new_ds

def collect_preprocess_data(train_dataset:str, adapted_neogate_dev:str, workflow:str, language:str):
    """
    Args:
      train_dataset: Path to the folder containing the training dataset. Files can be obtained by running:
        python scripts/create_my_dataset.py [your Hugging Face token]
      adapted_neogate_dev: Path to the dev split of the version of Neo-GATE adapted to our paradigm (obtained by following https://huggingface.co/datasets/FBK-MT/Neo-GATE#adaptation)
      workflow: Either "llm" or "seq2seq", depending on the model to train. Task prefixes (add_prefix()) and sentinel tokens (add_sentinel()) are added for seq2seq, simple template (add_template()) is applied for "llm" (details in paper). 

    Returns:
      train: Training data in Hugging Face format.
      val: Validation data in Hugging Face format. 
    """

    # Load training + validation data
    train_df = pd.read_csv(f"{train_dataset}/train.tsv", sep="\t")
    val_df = pd.read_csv(f"{train_dataset}/val.tsv", sep="\t")

    train = Dataset.from_pandas(train_df).shuffle(seed=42).flatten_indices() # Training split is shuffled
    val = Dataset.from_pandas(val_df)

    # Add Neo-GATE dev set (100 sentences) to validation set
    neogate_dev_schwa = [l.strip() for l in open(f"{adapted_neogate_dev}", encoding="utf-8").readlines()] # Sentences from the version of Neo-GATE dev set adapted to specific DNL paradigm
    neogate_dev = load_dataset("FBK-MT/Neo-GATE")["dev"].add_column(name="SCHWA", column=neogate_dev_schwa).to_pandas()
    neogate_dev_ds = Dataset.from_pandas(balance_neogate(neogate_dev))
    val = concatenate_datasets([val, neogate_dev_ds]).shuffle(seed=42).flatten_indices()

    if workflow == "seq2seq":
        # Add task-specific prefix to input sentences
        train = add_prefix(train, "REF", language)
        val = add_prefix(val, "REF", language)
        
        # Add sentinel token
        train = add_sentinel(train, "REF", "SCHWA")
        val = add_sentinel(val, "REF", "SCHWA")
    
    elif workflow == "llm":
        # Concatenate inputs and targets, separated by template
        train = add_template(train, "REF", "SCHWA", language)
        val = add_template(val, "REF", "SCHWA", language)

    return train, val

def load_model(model_path:str, workflow:str, qlora:bool):
    """
    Load the right class for the model, and get QLoRA version if requested. 

    Args:
      model_path: Path to the Hugging Face model to train. 
      workflow: "llm" loads AutoModelForCausalLM, "seq2seq" loads AutoModelForSeq2Seq.
      qlora: If the model has to be trained with QLoRA or not. 
    Returns:
      tokenizer: The model's tokenizer.
      model: The model, ready for QLoRA training if requested. 
    """
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")

    # Instantiate appropriate AutoModel class
    if workflow == "llm":
        model_class = AutoModelForCausalLM
    elif workflow == "seq2seq":
        model_class = AutoModelForSeq2SeqLM


    if qlora == True:
        # Set Quantization configuration
        q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
        )

        model = model_class.from_pretrained(
            model_path,
            quantization_config = q_config
        )
        
        model = prepare_model_for_kbit_training(model)

        # Set LoRA configuration
        target_modules = target_modules_map[f"{model.config.model_type}"] # Recommended target layers (Attention) from the peft library for each model

        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none"
        )

        model = get_peft_model(model, lora_config)

    elif qlora == False:
        model = model_class.from_pretrained(model_path)

    return tokenizer, model

def main(args):
    # Set model name for logging #
    model_name = args.model_path.split("/")[1]
    finetuned_model_name = f"{model_name}_qlora" if args.qlora == True else f"{model_name}_full"

    # Determine language #
    if args.language == None:
        language = "it" if "it" in model_name else "en"
    else:
        language = args.language

    # Load + preprocess data #
    train, val = collect_preprocess_data(args.dataset, args.adapted_neogate_dev, args.workflow, language)

    # Load + prepare prepare tokenizer & model #
    tokenizer, model = load_model(args.model_path, args.workflow, args.qlora)

    # Workflow-specific section #
    # Standard decoder-only LLMs
    if args.workflow == "llm":

        # Tokenize concatenated input
        train_tokenized = train.map(
            tokenize_simple,
            input_columns=["REF"],
            fn_kwargs={"tokenizer": tokenizer}
        )
        val_tokenized = val.map(
            tokenize_simple,
            input_columns=["REF"],
            fn_kwargs={"tokenizer": tokenizer}
        )

        # Instantiate appropriate classes
        data_collator_class = DataCollatorForLanguageModeling
        training_args_class = TrainingArguments
        trainer_class = Trainer

        # Set workflow-specific argument for data collator
        data_collator_kwargs = {
            "mlm": False
        }

    # Encoder-decoder (seq2seq) models
    elif args.workflow == "seq2seq":

        # Tokenize both inputs and labels
        train_tokenized = train.map(
            tokenize_both,
            input_columns=["REF", "SCHWA"],
            fn_kwargs={"tokenizer": tokenizer}
        )
        val_tokenized = val.map(
            tokenize_both,
            input_columns=["REF", "SCHWA"],
            fn_kwargs={"tokenizer": tokenizer}
        )

        # Instantiate appropriate classes
        data_collator_class = DataCollatorForSeq2Seq
        training_args_class = Seq2SeqTrainingArguments
        trainer_class = Seq2SeqTrainer

        # Set workflow-specific argument for data collator
        data_collator_kwargs = {
            "model": model
        }
    
    # Define common training arguments #
    training_args = {
        "output_dir": f"./trainers/{finetuned_model_name}",
        "overwrite_output_dir": True,
        "eval_strategy": "steps",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "logging_dir": f"./tensorboard/{finetuned_model_name}",
        "logging_strategy": "steps",
        "logging_steps": args.steps, # eval_steps also defaults to this
        "save_steps": args.steps,
        "label_names": ["labels"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "tensorboard",
        "group_by_length": True, # Create batches based on inputs of similar length to minimize padding
        "fp16": args.mixed_precision # Enable mixed precision
    }

    # Instantiate appropriate data collator #
    data_collator = data_collator_class(
        tokenizer=tokenizer,
        return_tensors="pt",
        **data_collator_kwargs
    )

    # Define common trainer parameters #
    trainer_args = {
        "args": training_args_class(**training_args),
        "model": model,
        "data_collator": data_collator,
        "train_dataset": train_tokenized,
        "eval_dataset": val_tokenized,
        "processing_class": tokenizer,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    }

    # Instantiate appropriate Trainer
    trainer = trainer_class(**trainer_args)

    # Train
    trainer.train()

    # Export model weights
    trainer.save_model(f"./models/{finetuned_model_name}")

    print(f"Model saved at ./models/{finetuned_model_name}\n" + \
          f"Find training report in folder ./tensorboard/{finetuned_model_name}\n" + \
            f"Trainer state saved in folder ./trainers/{finetuned_model_name}\n")
    
    return None

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path", 
        help="The path to the HuggingFace repository containing the model"
    )
    parser.add_argument(
        "workflow", 
        choices=["llm", "seq2seq"], 
        help="The type of model architcture; either 'llm' or 'seq2seq'"
    )
    parser.add_argument(
        "--dataset", 
        default="./dataset",
        help="The path to the training dataset. Defaults to ./dataset"
    )
    parser.add_argument(
        "--adapted_neogate_dev", 
        default="./adapted_neogate/adapted_neogate_dev.ref", 
        help="The path to the sentences from the version of Neo-GATE dev set adapted to specific paradigm. Defaults to ./adapted_neogate/adapted_neogate_dev.ref"
    )
    parser.add_argument(
        "--language",
        choices=["en", "it"], 
        default=None, 
        help="The language to use when writing prompts; either 'en' or 'it'. If not set, will be set to 'it' for models that have 'it' in their name, 'en' otherwise."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default="2e-4", 
        help="The learning rate for fine-tuning. Defaults to 2e-4."
    )
    parser.add_argument(
        "--qlora", 
        action="store_true", 
        default=False, 
        help="Add this flag to train using QLoRA. Defaults to False."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2, 
        help="The number of inputs/labels in each batch for training. Defaults to 2."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=200, 
        help="The number of steps (number of training examples / batch size) after which evaluation, logging, and checkpointing are performed. Defaults to 200."
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=2, 
        help="If eval metric does not improve after this number of logging steps, training will halt. Defaults to 2."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="The number of training epochs (complete iterations over training dataset). Defaults to 3."
    )

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli()