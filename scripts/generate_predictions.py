from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from datasets import Dataset, load_dataset
import pandas as pd
import argparse
import torch
import os

def load_model(model_path, workflow, quantization):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}")

    # Instantiate appropriate AutoModel class
    if workflow == "llm":
        model_class = AutoModelForCausalLM
    elif workflow == "seq2seq":
        model_class = AutoModelForSeq2SeqLM
    
    if quantization == True:
        # Set quantization configuration
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 # same as training
            )
        
        model = model_class.from_pretrained(
            model_path,
            quantization_config = q_config
        )
    else:
        model = model_class.from_pretrained(model_path)

    return tokenizer, model

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

def add_sentinel(ds:Dataset, ref_col:str) -> Dataset:
    new_refs = []

    for s in ds[f"{ref_col}"]:
        ref = f"{s.strip()} <extra_id_0>" # Append to input sentence
        new_refs.append(ref)

    new_ds = ds.remove_columns(f"{ref_col}").add_column(f"{ref_col}", new_refs)

    return new_ds

def add_template(ds:Dataset, ref_col:str, language:str) -> Dataset:
    refs = ds[f"{ref_col}"]
    
    templates = []
    for s in refs:
        if language == "it":
            template = f"Frase originale: <{s.strip()}> Riformulazione:"
        elif language == "en":
            template = f"Original sentence: <{s.strip()}> Rewritten sentence:"
        templates.append(template)

    new_ds = ds.remove_columns([f"{ref_col}"]).add_column(f"{ref_col}", templates)
    return new_ds

def preprocess_data(ref_col:str, workflow:str, prefix:bool, sentinel:bool, language:str) -> Dataset:
    neogate = load_dataset("FBK-MT/Neo-GATE")["test"].to_pandas()
    test = Dataset.from_pandas(balance_neogate(neogate))

    if workflow == "seq2seq":
        if prefix == True:  
            test = add_prefix(test, f"{ref_col}", language)

        if sentinel == True: # If prefix already added, will stay
            test = add_sentinel(test, f"{ref_col}")
    
    elif workflow == "llm":
        test = add_template(test, f"{ref_col}", language)
    
    return test

def get_predictions(tokenizer, model, test:list, ref_col:str) -> list:
    sents = test[f"{ref_col}"]
    predictions = []

    for i, sent in enumerate(sents):
        input_ids = tokenizer(
            sent,
            padding=False,
            truncation=False,
            return_tensors="pt"
        ).to(model.device).input_ids

        prompt_len = input_ids.shape[1]

        bos_token_id = model.config.bos_token_id if hasattr(model.config, "bos_token_id") else None
        decoder_start_token_id=model.config.decoder_start_token_id if hasattr(model.config, "decoder_start_token_id") else None

        gen_config = GenerationConfig(
            max_new_tokens = prompt_len,
            eos_token_id = model.config.eos_token_id,
            pad_token_id = model.config.pad_token_id,
            bos_token_id = bos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            stop_strings = ["</s>", ".", "<pad>"]
        )

        outputs = model.generate(
            tokenizer=tokenizer,
            inputs=input_ids,
            generation_config=gen_config
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        print(f"\r{i}/{len(sents)} predictions collected", end="", flush=True)
    print()

    return predictions

def write_predictions(predictions:list, logging_dir:str, save_model_name:str) -> None:
    with open(f"{logging_dir}/{save_model_name}_predictions.txt.clean", "a+", encoding="utf-8") as wf:
        for i, pred in enumerate(predictions):
            wf.write(f"{pred}\n")
        print(f"\r{i}/{len(predictions)} predictions written", end="", flush=True)
    print()

    print(f"Find outputs at {logging_dir}/{save_model_name}_predictions.txt.clean")

    return None

def main(args):
    model_name = args.model_path.split("/")[1]
    
    # Set parameters
    if args.quantization == None:
        quantization = True if "qlora" in model_name else False
    else:
        quantization = args.quantization

    if args.language == None:
        language = "it" if "it" in model_name else "en"
    else:
        language = args.language
    
    # Load tokenizer & model
    tokenizer, model = load_model(args.model_path, args.workflow, quantization)

    # Load evaluation data
    test = preprocess_data("REF", args.workflow, args.prefix, args.sentinel, language)

    # Collect predictions
    predictions = get_predictions(tokenizer, model, test, "REF")

    # Create folder + file name
    if args.prefix == True:
        save_model_name = f"{model_name}_prefix"
    if args.sentinel == True:
        save_model_name = f"{model_name}_sentinel"

    os.makedirs(f"{args.logging_dir}", exist_ok=True)

    # Write predictions to file
    write_predictions(predictions, args.logging_dir, save_model_name)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "workflow",
        choices=["seq2seq", "llm"],
        help="Type of model, either 'seq2seq' or 'llm'"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language for the prefix token. Defaults to 'it' if 'it' is in model name, 'en' otherwise"
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        default=None,
        help="This specifies whether the model to call was not fine-tuned in full precision. Defaults to True if 'qlora' is in the model name, False otherwise"
    )
    parser.add_argument(
        "--prefix",
        action="store_true",
        default=False,
        help="Add this flag to append a task-specific prefix to input sentences. Defaults to False"
    )
    parser.add_argument(
        "--sentinel",
        action="store_true",
        default=False,
        help="Add this flag to add sentinel tokens to inputs and labels. Defaults to False"
    )
    parser.add_argument(
        "--logging_dir",
        default="./results/finetuning",
        help="The folder to use for logging. Defaults to ./results/finetuning"
    )

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli()