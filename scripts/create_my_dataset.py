from argparse import ArgumentParser
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

parser = ArgumentParser()
parser.add_argument("hf_token", type=str, help="HuggingFace token to access gated repositories")
args = parser.parse_args()
access_token = args.hf_token

# mGeNTE (https://huggingface.co/datasets/FBK-MT/mGeNTE) #
# Load Set-N
mgente_setn = load_dataset("FBK-MT/mGeNTE", "mGeNTE en-it", token=access_token)["test"].filter(lambda x: x["SET"] == "Set-N")

# Collect Schwa refs
mgente_schwa = pd.read_csv("./data/mgente/add/mGeNTE_SetN_Schwa.tsv", sep="\t")["SENTENCE"].tolist()

# Add Schwa refs to Set-N
mgente_setn_schwa = mgente_setn.add_column("SCHWA", mgente_schwa)

# MT-GenEval (https://huggingface.co/datasets/gsarti/mt_geneval) #
# Load Context set
context = load_dataset("gsarti/mt_geneval", "context_en_it")

# Add features
def augment_context(ds:Dataset) -> Dataset:
    """
    This function is used to add our gender-neutral schwa sentences to the existing gender-marked translations found in the original MT-GenEval (Contextual) dataset. 
    It also adds a column indicating the gender of the original source sentence (M or F). 
    Args:
      ds: The original Hugging Face dataset with both splits (https://huggingface.co/datasets/gsarti/mt_geneval). 
    Returns:
      new_ds: A new version of the dataset (with both splits, now called "train" and "test") containing the reformulations (SCHWA column) and information about the original source sentences (GENDER). 
    """
    new_splits = {}

    for datasplit in ds.keys(): # Select train or test data

        # Read our files with features to augment original dataset
        # For each feature (schwa and gender), there is one file per split (train and dev)
        features = pd.read_csv(f"data/geneval/add/{datasplit}.tsv", sep="\t")
        gender = features["GENDER_ORIG"]
        schwa = features["SCHWA"]
    
        # Add schwa & gender columns + add corpus-specific ID
        new_splits[datasplit] = ds[datasplit].add_column("GENDER", gender).add_column("SCHWA", schwa).map(lambda x: {"orig_id": f"geneval_{datasplit}_{x['orig_id']}"})
    
    # Put splits together
    new_ds = concatenate_datasets([new_splits["train"], new_splits["test"]])
    
    return new_ds

context_schwa = augment_context(context)

# Balance gender
def balance_context(ds:Dataset) -> Dataset:
    """
    This function is used to create a version of MT-GenEval (Contextual set) which only contains one translation for each row. In order for the result to be gender-balanced, the feminine translation is collected from the first half of the dataset, and the masculine translation from the second half.

    Args:
      ds: The Hugging Face dataset with added GENDER column. This can be obtained by calling augment() on the original dataset. 

    Returns:
      new_ds: The new balanced dataset (same number of rows). 
    """
    sorted = ds.sort("GENDER") # Sort by gender (alphabetically: first all F, then all M)
    
    f = sorted.filter(lambda x: x["GENDER"] == "F") # Collect F sents
    m = sorted.filter(lambda x: x["GENDER"] == "M") # Collect M sents

    splitpoint = (sorted.num_rows) // 2

    f_from_f = f.map(lambda x: {"REF-G": x["reference_original"], "GENDER": "F"})   # Collect F translations from F half, add GENDER column
    f_from_m = m.map(lambda x: {"REF-G": x["reference_flipped"], "GENDER": "F"})    # Collect F translations from M half, add GENDER column
    f_combined = concatenate_datasets([f_from_f, f_from_m]) # Concatenate all F translations
    fdef = f_combined.select(range(splitpoint))             # but keep only half

    # Same for M translations
    m_from_f = f.map(lambda x: {"REF-G": x["reference_flipped"], "GENDER": "M"})  # from F half
    m_from_m = m.map(lambda x: {"REF-G": x["reference_original"], "GENDER": "M"}) # from M half
    m_combined = concatenate_datasets([m_from_f, m_from_m])
    mdef = m_combined.select(range(splitpoint, m_combined.num_rows))

    # Combine F and M translations
    new_ds = concatenate_datasets([fdef, mdef])

    return new_ds.sort("orig_id") # Restore original sorting

context_schwa_balanced = balance_context(context_schwa)

# Remove unused columns, Remove rows for which no schwa translation is available (details in paper)
context_def = context_schwa_balanced.remove_columns(["reference_original", "reference_flipped"]).filter(lambda x: x["SCHWA"] != "None")


# Put the two augmented datasets together #
# Rename columns for consistency, Remove unused columns, Add new columns
mgente_setn_schwa = mgente_setn_schwa.rename_columns({"Europarl_ID": "ORIG_ID"}).remove_columns(["ID", "SET", "COMMON", "GENDER", "REF-G_ann", "G-WORDS"]).\
    add_column("CORPUS", ["mGeNTE_Set-N"] * mgente_setn_schwa.num_rows).\
    add_column("GENDER", ["N"]*mgente_setn_schwa.num_rows) # All source sentences in mGeNTE Set-N are gender-Neutral

geneval_col = ["MT-GenEval_Context"] * context_def.num_rows
context_def = context_def.add_column("CORPUS", geneval_col).rename_columns({"orig_id": "ORIG_ID", "source": "SRC"}).remove_columns(["context"])

# Reorder columns for consistency before concatenation
final_columns = ["CORPUS", "ORIG_ID", "SRC", "REF-G", "GENDER", "SCHWA"]

def reorder(ds:Dataset, cols:list) -> Dataset:
    """
    Reorder columns of a Hugging Face dataset based on a list of column names.

    Args:
      ds: The original HF dataset.
      cols: The list of columns for the resulting dataset.
    Returns:
      new_ds: The new version of the same dataset, with the new column order. 
    """
    ds_new = ds.remove_columns([c for c in ds.column_names])
    for col in cols:
        if col not in ds_new.column_names:
            ds_new = ds_new.add_column(f"{col}", ds[f"{col}"])

    return ds_new

mgente_setn_schwa_reordered = reorder(mgente_setn_schwa, final_columns)
context_schwa_reordered = reorder(context_def, final_columns)

# Concatenate
combined = concatenate_datasets([mgente_setn_schwa_reordered, context_schwa_reordered]).shuffle() # Shuffle after concatenation

# Add new IDs after shuffling
combined_ids = combined.add_column("ID", [i+1 for i in range(combined.num_rows)])
def_columns = ["ID", "CORPUS", "ORIG_ID", "SRC", "REF-G", "GENDER", "SCHWA"] # Reorder again so that new ID is the first column
combined_def = reorder(combined_ids, def_columns)


# Create test split (10%), convert all to .tsv, and download #
dnl_rewriter_dataset = combined_def.train_test_split(test_size=0.1, shuffle=False) # Keep new IDs in place

dnl_rewriter_dataset["train"].to_csv("./dataset/train.tsv", sep="\t", na_rep=None, lineterminator="\n")
dnl_rewriter_dataset["test"].to_csv("./dataset/val.tsv", sep="\t", na_rep=None, lineterminator="\n")