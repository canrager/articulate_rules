#%% Imports and setup
import re
import json
import numpy as np
import itertools
from datasets import load_dataset
from tqdm import tqdm
import openai

with open("openaik.txt", "r") as file:
    openai.api_key = file.readline().strip()

#%% Download datasets
c4_dataset = load_dataset("NeelNanda/c4-10k", split="train")
wiki_dataset = np.array(load_dataset("wikipedia", "20220301.simple", split="train")["text"][:10000])

#%% Global experiment variables
LEN_C4 = 10000
LEN_WIKI = 10000
LEN_DATASET = 100
N_MIN_INPUT_CHAR = 50
DATASET_DIR = "./data/n100-loaded/"

# %% Load Lowercase
def load_lowercase():
    # Rule definiton
    rule_title = "lowercase_same"
    rule_formulation_true = "The input only contains characters in lowercase."
    rule_formulation_false = "The input contains at least one uppercase letter."
    
    # Variables
    ds = dict(
        rule_title=rule_title,
        rule_formulation_true=rule_formulation_true,
        rule_formulation_false=rule_formulation_false,
        true_samples=[],
        false_samples=[],
    )

    # True Inputs
    true_idxs = np.random.choice(LEN_C4, LEN_DATASET)
    # ds["true_samples"] = [s.split(".")[0].lower() for s in c4_dataset[true_idxs]["text"]]
    ds["true_samples"] = [s.split(".")[0].lower() for s in wiki_dataset[true_idxs]]

    # Same false inputs 
    # ds["false_samples"] = [s.split(".")[0] for s in c4_dataset[true_idxs]["text"]]
    ds["false_samples"] = [s.split(".")[0] for s in wiki_dataset[true_idxs]]

    # Random False Inputs
    # false_cnt = 0
    # while false_cnt < LEN_DATASET:
    #     idx = np.random.randint(LEN_C4)
    #     s = c4_dataset[idx]["text"].split(".")[0]
    #     if bool(re.match(r'\w*[A-Z]\w*', s)):
    #         ds['false_samples'].append(s)
    #         false_cnt += 1
    
    with open(DATASET_DIR + f"{rule_title}.json", "w") as file:
        json.dump(ds, file)

load_lowercase()
#%% YEARS 
def preprocess_wiki(sample):
    sample = re.sub(r'[^\x00-\x7f]',r'-', sample) #remove non-ascii characters
    sample = sample.split("\n")
    sample = [p.split(".") for p in sample]
    sample = itertools.chain(*sample)
    sample = [s.strip(' .,:-_;*()"0123456789\n') for s in sample] # remove numbering or other formatting
    return sample



def load_years():
    # Rule definiton
    rule_title = "years"
    rule_formulation_true = "The input contains at least one four-digit year."
    rule_formulation_false = "The input contains no four-digit year."
    
    # Variables
    ds = dict(
        rule_title=rule_title,
        rule_formulation_true=rule_formulation_true,
        rule_formulation_false=rule_formulation_false,
        true_samples=[],
        false_samples=[],
    )
    reg_exp = r'(?:[0-9]{4})'


    # True Inputs
    cnt = 0
    while cnt < LEN_DATASET:
        idx = np.random.randint(LEN_WIKI)
        inputs = preprocess_wiki(wiki_dataset["text"][idx])
        for s in inputs: # take first sentence with 4 digit number
            if bool(re.search(reg_exp, s)) and len(s) > N_MIN_INPUT_CHAR:
                ds['true_samples'].append(s)
                cnt += 1
                print(cnt, s)
                break

    # False Inputs
    cnt = 0
    while cnt < LEN_DATASET:
        idx = np.random.randint(LEN_WIKI)
        inputs = preprocess_wiki(wiki_dataset["text"][idx])
        for s in inputs: # take first sentence with 4 digit number
            if re.search(reg_exp, s) == None and len(s) > N_MIN_INPUT_CHAR:
                ds['false_samples'].append(s)
                cnt += 1
                print(cnt, s)
                break
    
    with open(DATASET_DIR + f"{rule_title}.json", "w") as file:
        json.dump(ds, file)

#load_years()

# %% German
def load_german():
    # Rule definiton
    rule_title = "german"
    rule_formulation_true = "The input is written in German language."
    rule_formulation_false = "The input is written in French language"
    
    # Variables
    ds = dict(
        rule_title=rule_title,
        rule_formulation_true=rule_formulation_true,
        rule_formulation_false=rule_formulation_false,
        true_samples=[],
        false_samples=[],
    )
    
    
    true_idxs = np.random.choice(LEN_C4, LEN_DATASET)
    # inputs = [s.split(".")[0] for s in c4_dataset[true_idxs]["text"]]
    inputs = [s.split(".")[0] for s in wiki_dataset[true_idxs]]
    for s in tqdm(inputs):
        # True inputs
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Please translate the following sentence to German, exactly retaining its content: "+s}
        ]
        out = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        ds["true_samples"].append(out.choices[0].message.content)

        # False inputs
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Please translate the following sentence to French, exactly retaining its content: "+s}
        ]
        out = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        ds["false_samples"].append(out.choices[0].message.content)

    
    with open(DATASET_DIR + f"{rule_title}.json", "w") as file:
        json.dump(ds, file)

load_german()

# %% Generalized labeling with GPT 3.5

def label_ds(rule_title, rule_formulation_true, rule_formulation_false):
    classification_prompt = f'Please label the following input as "True" or "False" given the following rule.\n Rule: {rule_formulation_true}\n'
    
    # Variables
    ds = dict(
        rule_title=rule_title,
        rule_formulation_true=rule_formulation_true,
        rule_formulation_false=rule_formulation_false,
        true_samples=[],
        false_samples=[],
    )
    
    true_cnt = 0
    false_cnt = 0
    while true_cnt < LEN_DATASET or false_cnt < LEN_DATASET:
        # Load random input
        idx = np.random.randint(LEN_C4)
        s = c4_dataset[idx]["text"].split(".")[0]

        # Label input
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": classification_prompt + f'This is the input: ' + s}
        ]

        # Let GPT generate the dataset samples
        out = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        text = out.choices[0].message.content

        # Save input
        if text == "True" and true_cnt < LEN_DATASET:
            ds["true_samples"].append(s)
            true_cnt += 1
            # print(f'True: {s}')
        elif text == "False":
            if false_cnt < LEN_DATASET:
            #     ds["false_samples"].append(s)
            #     false_cnt += 1
            #     # print(f'False: {s}')
            # else:
                rephrasing_prompt = f"Please come up with a new input inspired by the content of sentence A. The new input should follow this rule:\n"
                # Label input
                messages = [
                    {"role": "system", "content": "You are a assistant who only outputs a single sentence and speaks a neutral tone."},
                    {"role": "user", "content": rephrasing_prompt + rule_formulation_true + '\nThis is sentence A: ' + s}
                ]

                # Let GPT generate the dataset samples
                out = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=messages,
                )
                text = out.choices[0].message.content
                ds["true_samples"].append(text)
                true_cnt += 1
                print(f'inspiration {s}')
                print(f'New generated true input: {text}')
                print("\n\n")
        else:
            print(text)
        print(f'progress {(false_cnt + true_cnt) / LEN_DATASET / 2}')
    
    with open(DATASET_DIR + f"{rule_title}.json", "w") as file:
        json.dump(ds, file)


label_ds(
    rule_title="color",
    rule_formulation_true="The input contains a color.",
    rule_formulation_false="The input does not contain a color.",
)
# %% Generalized dataset creation

"""
We have a corpus of random web text.
We have a rule for binary classification: True / False
Goal: A dataset with two classes. The classification feature itself should highly correlate with its label, while all other features should not correlate in or across samples.

Problem: random samples of web text may be heavily biased towards not containing the classification feature at all.
Problem: generating random samples based on a simple rule often shows high correlation with other, non-relevant features.



"""
