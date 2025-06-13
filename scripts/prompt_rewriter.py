import argparse
import os
import torch
import pandas as pd
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers.modeling_outputs import BaseModelOutput

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

def collect_preprocess_data(train_data:str, val_data:str, adapted_neogate_dev:str):
    """
    Load data from files obtained by running scripts/create_my_dataset. Add the dev split of the adapted Neo-GATE to the validation data. 

    Args:
      train_data: Path to the training data (file obtained by running create_my_dataset.py).
      val_data: Path to the validation data (file obtained by running create_my_dataset.py).
      adapted_neogate_dev: Path to the adapted version of Neo-GATE dev split (obtained by following https://huggingface.co/datasets/FBK-MT/Neo-GATE#adaptation). 
    Returns:
      train: The training set, including validation data (including Neo-GATE dev).
      test: The test set (Neo-GATE test, loaded from https://huggingface.co/datasets/FBK-MT/Neo-GATE).
    """
    train = Dataset.from_pandas(pd.read_csv(f"{train_data}", sep="\t"))
    val = Dataset.from_pandas(pd.read_csv(f"{val_data}", sep="\t"))

    # Add my version of Neo-GATE dev split to training data
    adapted_ref_dev = [l.strip() for l in open(f"{adapted_neogate_dev}", encoding="utf-8").readlines()] # Neo-GATE references adapted to my paradigm
    neogate_dev_schwa = load_dataset("FBK-MT/Neo-GATE")["dev"].add_column(name="SCHWA", column=adapted_ref_dev).to_pandas()
    neogate_dev_balanced = Dataset.from_pandas(balance_neogate(neogate_dev_schwa))

    # Concatenate training set
    train = concatenate_datasets([train, val, neogate_dev_balanced]) # Shuffling happens when prompts are created

    # Load test set
    neogate = load_dataset("FBK-MT/Neo-GATE")["test"].to_pandas()
    test = Dataset.from_pandas(balance_neogate(neogate)) # Gender-balance this split too

    return train, test

def load_model(model_path:str, access_token:str, quantization:int, workflow:str):
    """
    Load correct class for tokenizer and model; model is quantized if necessary. 

    Args:
      model_path: Path to the Hugging Face folder of the model to prompt. 
      access_token: Hugging Face token, necessary if model is gated. 
      quantization: If model should be quantized or not.
      workflow: Either "llm" or "seq2seq". "llm" loads AutoModeForCausalLM, "seq2seq" loads AutoModelForSeq2Seq.
    Returns:
      tokenizer: The model's tokenizer.
      model: The model, quantized if requested, otherwise as stored in the source repo. 
    """
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", token=access_token)

    model_args = {
        "pretrained_model_name_or_path": f"{model_path}",
        "token": f"{access_token}"
    }

    # Set quantization parameters
    if quantization is not None:
        if quantization == 4:
            q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
            )
        
        elif quantization == 8:
            q_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        model_args["quantization_config"] = q_config
        model_args["device_map"] = "auto"

    # Select appropriate AutoModel class based on model type
    if workflow == "llm":
        model_class = AutoModelForCausalLM
    elif workflow == "seq2seq":
        model_class = AutoModelForSeq2SeqLM

    model = model_class.from_pretrained(**model_args)

    print("Note: Model takes up {} bytes of memory".format(model.get_memory_footprint()))

    return tokenizer, model

def create_prompts(language:str, workflow:str, train_sources:list, train_targets:list, test:list, num_examples:int):
    """
    Create list of target inputs (requests) and batches of k (num_examples) examples separately. This function does NOT add a description of the task: to get that, call add_instructions() on the examples returned by this function. 

    Args:
        language: Either "it" or "en", the language to write the prompts. 
        workflow: Either "llm" or "seq2seq". "seq2seq" adds task prefixes and sentinel tokens (details in paper). 
        train_sources: The list of reference sentences (inputs) used to create example pairs.
        train_targets: The list of target sentences (labels) used to create example pairs.
        test: The list of test (input) sentences the model should rewrite.
        num_examples: The number of examples (shots) for each request.
    
    Returns:
        example_batches (list): A list containing a batch of k example pairs for each request.
        target_inputs (list): A list containing target inputs, i.e., requests to send the model. 
    """
    # Create open-ended target input (request)
    target_inputs = []
    for sent in test:
        sent = sent.strip()
        if language == "en":
            target_input = f"Original sentence: <{sent}> Rewritten sentence:"
        elif language == "it":
            target_input = f"Frase originale: <{sent}> Riformulazione:"
        
        if workflow == "seq2seq":
            target_input = f"{target_input} <extra_id_0>" # Add sentinel token
        else:
            continue
        
        target_inputs.append(target_input)

    # Create example set
    train_pairs = list(zip(train_sources, train_targets)) # Examples are made up of a reference traslation + schwa reformulation

    train_examples = [] # Create example pairs by concatenating input and target
    i=0
    for pair in train_pairs:
        src = pair[0].strip()
        tgt = pair[1].strip()
        if language == "en":
            example_pair = f"Original sentence: <{src}> Rewritten sentence: {tgt}</s>"
        elif language == "it":
            example_pair = f"Frase originale: <{src}> Riformulazione: {tgt}</s>"
        train_examples.append(example_pair)
        i+=1

    # Create n batches of example pairs, where n is the number of requests (sentences in the test set)
    example_batches = []
    for i in target_inputs: # A batch is a set of k example pairs
        batch = random.sample(train_examples, num_examples)
        example_set = "\n".join(batch) # Add a newline between each example pair and the next
        example_batches.append(example_set)

    return example_batches, target_inputs

def create_messages(train_sources:list, train_targets:list, test:list, num_examples:int) -> list:
    """
    Create a list of chat messages containing:
    - instructions and input examples from the user
    - sample assistant outputs
    - a request
    This is meant for "chat" models, whose configuration has a chat_template that the tokenizer can use (this is required).

    Args:
        train_sources: The list of reference sentences (inputs) used in the user input.
        train_targets: The list of target sentences (labels) used in the sample assistant output.
        test: The list of test (input) sentences used to send requests.
        num_examples: The number of examples (shots) for each request.
    
    Returns:
        messages (list): A list of chat messages.
    """
    messages = []

    # Create open-ended target input (request) with sentinel token
    target_inputs = []
    for sent in test:
        target_inputs.append(f"Original sentence: <{sent.strip()}> Rewritten sentence: ")

    if num_examples == 0:
        # Create 0-shot prompts
        for i, inpt in enumerate(target_inputs):
            message = [
                {
                    "role": "user",
                    "content": f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n{inpt}"
                }
            ]

            messages.append(message)

    else:
        # Create example set
        train = list(zip(train_sources, train_targets)) # Examples are made up of reference traslation + schwa reformulation

        example_pairs = []
        for pair in train: # Create templates for the example pairs
            src = pair[0].strip()
            tgt = pair[1].strip()
            example_source = f"Original sentence: <{src}> Rewritten sentence: "
            example_target = f"<{tgt}>"

            example_pairs.append((example_source, example_target))

        # Create complete messages with batces of k examples, where k = num_examples 
        for i in range(len(target_inputs)):
            example_batch = random.sample(example_pairs, k=num_examples)
            src_batch = []
            tgt_batch = []
            for pair in example_batch:
                src_batch.append(pair[0].strip())
                tgt_batch.append(pair[1].strip())
            target_input = target_inputs[i]
            message = [
                {
                    "role": "user",
                    "content": f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n" + "\n".join(src_batch)
                },
                {
                    "role": "assistant",
                    "content": "\n".join(tgt_batch)
                },
                {
                    "role": "user",
                    "content": f"{target_input}"
                }
            ]
            messages.append(message)
    
    return messages

def add_instructions(language:str, examples:list) -> list:
    """
    Enhance prompts for instruction-tuned models by opening the prompt with a description of the task. 
    This is the default for chat models, so this function can only be called on a dedicated list of examples, not on complete prompts. 
    This function is thus meant for standard (non-chat) decoder-only LLMs mainly. 
    It can also be used for prompting encoder-decoder (seq2seq) models, in which case the instructions will be later passed to the encoder together with the examples and the request (not our implementation). 

    Args:
        language: Either "en" or "it", the language to write instructions.
        examples: The list of task examples obtained by calling create_prompts. 
    Returns:
        istructions: New list of task examples, with added task instructions (one instruction for each example set). 
    """
    instructions = []
    for batch in examples:
        if language == "en":
            instr = f"Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n{batch}"
        elif language == "it":
            instr = f"Riscrivi la seguente frase italiana utilizzando uno schwa (ə) al posto delle desinenze maschili e femminili per i referenti umani.\n{batch}"
        instructions.append(instr)

    return instructions

def get_encoder_outputs(tokenizer, model, example_set:str, target_input:str) -> BaseModelOutput:
    """
    This function is used when prompting encoder-decoder (seq2seq) models with a fusion-based approach, where each example is processed separately by the encoder. 

    Args:
        tokenizer: The tokenizer to tokenize each example.
        model: The model to prompt.
        example_set: A batch of k examples (shots) separated by newline (\n).
        target_input: A request.
    Returns:
        encoder outputs: A tuple containing the concatenated encoder last hidden states and attention masks (one of each per example). 
    """
    encoder_outputs = []
    attention_masks = []
    for example in example_set.split("\n"): # Encode each example together with the target input
        encoder_inputs = tokenizer(
            f"{example}\n{target_input}",
            return_tensors = "pt"
        ).to(model.device)

        attention_masks.append(encoder_inputs.attention_mask)

        lhs = model.encoder(**encoder_inputs).last_hidden_state

        encoder_outputs.append(lhs)
    
    concat_lhs = BaseModelOutput(last_hidden_state=torch.cat(encoder_outputs, dim=1)) # Concatenate all hidden states and wrap as BaseModelOutput
    concat_am = torch.cat((attention_masks), dim=1)

    return concat_lhs, concat_am

def kshot_prompt(workflow:str, tokenizer, model, example_batches:list, target_inputs:list) -> list:
    """
    Send prompts and collect outputs (predictions). 
    This function can work with both decoder-only and encoder-decoder models, but it is only suitable for k-shot prompting. Use zeroshot_prompt() for 0-shot prompting. Use complete_chat() for chat models. 
    In the case of encoder-decoder models, this function adopts the "early fusion" approach (more details in paper) by calling get_encoder_outputs(). 
    The arguments for this function can be obtained by calling create_prompts(). 

    Args:
        workflow: Either "llm" or "seq2seq", based on the model in use.
        tokenizer: The tokenizer.
        model: The model.
        example_batches(list): A list containing batches of k newline-separated examples. There has to be one batch of examples for each request to send to the model. 
        target_inputs(list): A list containing requests to send to the model. A request contains an input sentence and it gives the model the prompt to generate its output. 
    
    Returns:
        predictions: A list of semi-post-processed model outputs (one per request). 
    """
    predictions = []

    zipped = list(zip(example_batches, target_inputs)) # Create tuples containing a set of examples and a request
    for i, item in enumerate(zipped):
        example_set = item[0]  # Examples
        target_input = item[1] # Request

        if workflow == "seq2seq":
            encoder_hidden_states, encoder_attention_masks = get_encoder_outputs(tokenizer, model, example_set, target_input) # Get encoder hidden states for examples and request
            
            input_len = tokenizer(target_input, return_tensors="pt").input_ids.shape[1]

            decoder_inputs = tokenizer(
                "<extra_id_0>",
                return_tensors="pt"
            ).to(model.device).input_ids # Use the sentinel token to prompt the decoder to generate the rewritten sentence

            output = model.generate(
                tokenizer = tokenizer,
                inputs = decoder_inputs,
                attention_mask = encoder_attention_masks,
                encoder_outputs = encoder_hidden_states, # Use concatenated encoder last hidden states for decoder cross-attention
                max_new_tokens = input_len,
                stop_strings = ["."]
                )
        
        elif workflow == "llm":
            prompt = f"{example_set}\n{target_input}" # Concatenate examples and request with a newline in between
            
            prompt_input_ids = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(model.device).input_ids

            prompt_len = prompt_input_ids.shape[1]

            output = model.generate(
                inputs = prompt_input_ids,
                max_new_tokens = prompt_len
            )

        prediction = tokenizer.decode(output[0], skip_special_tokens=True) # Decode the logits
            
        predictions.append(prediction.split("\n")[-1]) # Try to only collect final answer if the model outputs the whole prompt
        print(f"\r{i+1}/{len(zipped)} predictions collected", end="", flush=True)
    print()
    
    return predictions

def complete_chat(tokenizer, model, messages:list) -> list:
    """
    Send chat messages to chat models. 
    The arguments for this function can be obtained by calling create_messages().
    It is only suitable for k-shot prompting; use no_examples() for 0-shot prompting.

    Args:
        tokenizer: The tokenizer.
        model: The model.
        messages: The list of chat prompts containing a set of k examples and a request. 
    Returns:
        predictions (list): A list of raw model outputs (one per request). 
    """
    predictions = []

    for i, message in enumerate(messages): # Tokenize by applying model-specific chat template
        message_tokenized = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        message_len = message_tokenized.input_ids.shape[1]

        outputs = model.generate(
            inputs=message_tokenized.input_ids,
            attention_mask=message_tokenized.attention_mask,
            max_new_tokens=message_len,
            pad_token_id = tokenizer.eos_token_id
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction) # NOTE: The collected prediction will need post-processing
        print(f"\r{i+1}/{len(messages)} predictions collected", end="", flush=True)
    print()

    return predictions

def zeroshot_prompt(tokenizer, model, language:str, test:list) -> list:
    """
    This function is used for 0-shot prompting; it compiles requests and adds explicit instructions to each request sent. It is only meant for standard (non-chat) decoder-only models when no examples (training data) are provided. It should not be called in combination with add_instructions().

    Args:
        tokenizer: The tokenizer, returned by load_model().
        model: The model to prompt, returned by load_model().
        language: Either "en" or "it", the language to write prompts. 
        test: A list of requests to send to the model. This can be obtained by calling create_prompts(). 
    
    Returns:
        predictions: A list of semi-post-processed model outputs (one per request).
    """
    target_inputs = []
    for sent in test:
        if language == "en":
            target_inputs.append(f"Original sentence: <{sent.strip()}> Rewritten sentence:")
        elif language == "it":
            target_inputs.append(f"Frase originale: <{sent.strip()}> Riformulazione:")

    predictions = []
    
    for i, inpt in enumerate(target_inputs):
        prompt = "Rewrite the following Italian sentence by replacing masculine and feminine endings with a schwa (ə) for human entities.\n" + inpt

        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device).input_ids
    
        prompt_len = input_ids.shape[1]

        outputs = model.generate(
            inputs = input_ids,
            max_new_tokens = prompt_len
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction.split("\n")[-1]) # Try to only collect final answer
        print(f"\r{i+1}/{len(target_inputs)} predictions collected", end="", flush=True)
    print()

    return predictions

def write_predictions(save_model_name:str, logging_dir:str, preds:list) -> None:
    """
    Write model outputs to a file, where each output will be on one line. No post-processing is applied at this stage. 

    Args:
        save_model_name: The name of the model that will appear in the file containing predictions. 
        logging_dir: The directory where to save the predictions. 
        preds: The list of model predictions obtained by calling one of get_predictions(), complete_chat(), or no_examples(). 
    """
    i = 1
    with open(f"{logging_dir}/{save_model_name}_predictions.txt", "a+", encoding="utf-8") as wf:
        for pred in preds:
            wf.write(f"{pred}\n")
            print(f"\r{i}/{len(preds)} model outputs written", end="", flush=True)
            i+=1
        print()
        print(f"Find outputs at {logging_dir}/{save_model_name}_predictions.txt")
    
    return None

def main(args):
    # Get model name
    model_name = args.model_path.split("/")[1]

    # Infer language
    if args.lang == None:
        language = "it" if "it" in model_name else "en"
    else:
        language = args.lang

    # Create logging dir
    os.makedirs(f"{args.logging_dir}", exist_ok=True)
    
    save_model_name = f"{model_name}_{language}"

    # Quantize
    if args.quantization is not None:
        if args.quantization == 4:
            save_model_name = f"{save_model_name}_4bit"
        elif args.quantization == 8:
            save_model_name = f"{save_model_name}_8bit"
    else:
        save_model_name = f"{save_model_name}_full"
    
    # Add number of shot to model name for log files
    save_model_name = f"{save_model_name}_{args.num_examples}shot"

    # Load data
    train, test = collect_preprocess_data(args.train_path, args.val_path, args.adapted_neogate_dev)

    inputs = train["REF"]
    labels = train["SCHWA"]

    # Load tokenizer and model
    tokenizer, model = load_model(f"{args.model_path}", args.hf_token, args.quantization, args.framework)

    # Shorten each individual input/label that will be used to create examples, if necessary to save memory
    # This way, only examples will be shortened, leaving the instructions and request intact
    if args.example_maxlen != None:
        inputs = [" ".join(inpt.split(" ")[:args.example_maxlen]) for inpt in inputs]
        labels = [" ".join(labl.split(" ")[:args.example_maxlen]) for labl in labels]

    # Create + send prompts
    if args.chat == True:
        messages = create_messages(train["REF"], train["SCHWA"], test["REF"], args.num_examples)
        predictions = complete_chat(tokenizer, model, messages)
    else:
        if args.num_examples == 0:
            predictions = zeroshot_prompt(tokenizer, model, language, test["REF"])
        else:
            examples, target_inputs = create_prompts(language, args.workflow, train["REF"], train["SCHWA"], test["REF"], args.num_examples)
            if args.instructions == True:
                examples = add_instructions(language, examples)
                save_model_name = f"{save_model_name}_instructions"
            predictions = kshot_prompt(args.workflow, tokenizer, model, examples, target_inputs)
    
    # Write model outputs to file
    write_predictions(save_model_name, args.logging_dir, predictions)
    print()

    return None

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        help="The path to the Hugging Face model to prompt, formatted as 'author/model'"
    )
    parser.add_argument(
        "framework",
        choices=["llm", "seq2seq"],
        help="The type of model to train; either 'llm' or 'seq2seq'"
    )
    parser.add_argument(
        "--train_path", 
        default="./dataset/train.tsv",
        help="The path to the training data"
    )
    parser.add_argument(
        "--val_path", 
        default="./dataset/val.tsv",
        help="The path to the validation data"
    )
    parser.add_argument(
        "--adapted_neogate_dev",
        default="./adapted_neogate/adapted_neogate_dev.ref",
        help="The path to the sentences from the version of Neo-GATE dev set adapted to your paradigm. Defaults to ./data/adapted_neogate/adapted_neogate_dev.ref"
    )
    parser.add_argument(
        "--logging_dir",
        default="./results/prompting", 
        help="The path to the folder to use for logging"
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token to access gated repositories, if needed. Defaults to None."
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="The language to use for prompts. Defaults to 'it' if 'it' is in model name, 'en' otherwise"
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=None,
        help="Specify the precision to quantize the model before calling it, if necessary. Either 4 or 8 (for 4-bit or 8-bit precision)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=2,
        help="The number of examples to include in each prompt. If set to 0, switches to 0-zhot prompting. Defaults to 2"
    )
    parser.add_argument(
        "--example_maxlen",
        type=int,
        default=None,
        help="If set, inputs and labels will be truncated to this number of tokens when creating example pairs."
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        default=False,
        help="Add this flag to add task instructions at the beginning of each prompt. Defaults to False"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=False,
        help="Add this flag if calling a chat model"
    )

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli()