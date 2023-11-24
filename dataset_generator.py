# %%
import openai
import json

# %% Set Keys
with open("openaik.txt", "r") as file:
    openai.api_key = file.readline().strip()

#%%
rules = {
    "lowercase": "The input only contains characters in lowercase.",
    "german": "The input is written in german.",
    "dates_before_200": "The input only contains dates before the year 2000.",
    "colors": "The input mentions a color.",
    "positive_sentiment": "The input expresses a positive sentiment.",
    "political_left": "The input statement is in favor of the ideals of the US democratic party.",
    "capitalistic": "The input statement is in favor of capitalistic ideals.",
    "palindrome": "The input contains a palindrome",
    "consecutive_letters": "The input contains doubled letters at least once.",
    "subject_ends_with_c": "The subject of the input statement ends with the letter 'c'.",
    "female_subject": "The subject of the input statement is female."
}

with open(f"./outputs/rules.txt", "w") as file:
    json.dump(rules, file)

#%%

len_ds = 6
system_prompt = "You are a helpful assistant."

#%%
for rule_title in rules:
    rule_formulation = rules[rule_title]
    experiment_description = f'Please generate a dataset of {len_ds} inputs labeled as True or False given the following rule: {rule_formulation} {len_ds//2} sentences should be generated fulfilling the rule and labeled true. The other {len_ds//2} sentences should be generated explicitly not fulfilling the rule and labeled as false. Please explicitly use the following format:\n\nInput: the cat sat on the mat. Label: True; Input: THE DOG RAN IN THE PARK. Label: False; Input: The ice is cold. Label: False;'
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": experiment_description}
    ]

    # Let GPT generate the dataset samples
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    text = out.choices[0].message.content
    print(text)

    # Convert generated dataset to py dict
    if type(text) == list:
        print("Output is a list.")
    if type(text) == str:
        text = text.strip().replace("\n", ";")
        text = text.split(";")
    else:
        raise NotImplementedError("Unsupported return type.")

    dataset = dict()
    dataset["rule_formulation"] = rule_formulation
    dataset["samples"] = dict()
    for i in range(len_ds):
        inp, lbl = text[i].split("Label")
        inp = inp.split(":")[1].strip(' ":;,.')
        lbl = lbl.strip(' ":;,.')
        dataset["samples"][i] = {"input": inp, "label": lbl}

    # Save generated dataset
    with open(f"./outputs/{rule_title}.txt", "w") as file:
        json.dump(dataset, file)

# %%
