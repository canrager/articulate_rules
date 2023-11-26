#%% Setup
import json
import openai
import numpy as np

with open("openaik.txt", "r") as file:
    openai.api_key = file.readline().strip()

#%% Experiment variables

OUTPUT_DIR = "./data/n100/"
N_SAMPLES_PER_LABEL = 100
N_SAMPLES_PER_BATCH = 10
SYSTEN_PROMPT = "You are a helpful and creative assistant. We collect examples for sentence classification." #Each sentence should repect user privacy, promote peaceful behavior, only describe legal contexts, and promote diversity and inclusion."
TEMPERATURE = 1
# experiment_description = f'Please generate a dataset of {2*n_samples_per_class} sentences labeled as True or False given the following rule: {rule_formulation} The {n_samples_per_class} sentences in the first half should be generated fulfilling the rule and labeled True. The {n_samples_per_class} sentences in the second half should be generated with a random context not fulfilling the rule and labeled as False. Please explicitly use the following format:\n\nsentence: the cat sat on the mat. Label: True\nInput: THE DOG RAN IN THE PARK. Label: False\nInput: The ice is cold. Label: False'

rules = {
    "lowercase":            dict(true="The sentence only contains characters in lowercase.", false="The sentence contains at least one uppercase letter."),
    "german":               dict(true="The sentence is written in german", false="The sentence is written in a language other than german and does not contain an name of a language or country or city."),
    "dates_before_2000":    dict(true="The sentence only contains dates before the year 2000.", false="The sentence only contains dates after the year 2000."),
    "colors":               dict(true="The sentence mentions a color.", false="The sentence does not mention any color."),
    "positive_sentiment":   dict(true="The sentence expresses a positive sentiment.", false="The sentence expresses negative, neutral, or no sentiment."),
    "political_left":       dict(true="The sentence is in favor of the ideals of the US democratic party and does not contain the words 'democrat' or 'conservative' or 'social' or 'republican'.", false="The sentence contradicts the ideals of the US democratic party and does not contain the words 'democrat' or 'conservative' or 'social' or 'republican'."),
    "capitalistic":         dict(true="The sentence is in favor of capitalistic ideals and does not contain the words 'capitalist' or 'communist' or 'socialist'.", false="The sentence contradicts capitalistic ideals and does not contain the words 'capitalist' or 'communist' or 'socialist'."),
    "palindrome":           dict(true="The sentence contains a palindrome", false="The sentence does not contain a palindrome."),
    "consecutive_letters":  dict(true="The sentence contains doubled letters at least once.", false="The sentence contains no doubled letters."),
    "subject_ends_with_c":  dict(true="The subject of the sentence ends with the letter 'c'.", false="The subject of the sentence does not end with the letter c."),
    "female_subject":       dict(true="The subject of the sentence is female.", false="The subject of the sentence is male."),
}

with open(OUTPUT_DIR + f"rules.json", "w") as file:
    json.dump(rules, file)

#%% Helper Functions

def generate_ds(rule_title):
    rule_formulation = rules[rule_title]
    dataset = dict(
        rule_title=rule_title,
        rule_formulation=rule_formulation,
        true_samples=[],
        false_samples=[]
    )

    for case in ["true", "false"]:
        # Assemble Prompt for dataset generation
        rule_formulation = rules[rule_title][case]
        desc = f'Please give {N_SAMPLES_PER_BATCH} short examples of sentences classifying as True given the following rule: {rule_formulation} Be creative! The content should not unveil the rule easily. Each example should have a very differnet context, but all of them should fulfill the rule. The sentences should be separated by "\n". Please avoid numbering or bullet points. The examples begin here:' # The dataset should be explicitly formatted as: sentence1\nsentence2\nsentence3\nsentence4\nsentence5'
        messages = [
            {"role": "system", "content": SYSTEN_PROMPT},
            {"role": "user", "content": desc}
        ]

        # Generated samples get very repetitive, if samples are generated at once.
        # Samples should be generated in batches
        for _ in range(N_SAMPLES_PER_LABEL // N_SAMPLES_PER_BATCH):

            # Let GPT generate the dataset samples
            out = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                temperature=TEMPERATURE,
                seed=np.random.randint(1e6)
            )
            text = out.choices[0].message.content

            # Insert generated samples in dataset
            text = text.replace("I.", "").replace("II.", "").replace("III.", "").replace("IV.", "").replace("V.", "")
            text = text.replace("\n1.", "\n").replace("\n2.", "\n").replace("\n3.", "\n").replace("\n4.", "\n").replace("\n5.", "\n")
            text = text.replace("\n6.", "\n").replace("\n7.", "\n").replace("\n8.", "\n").replace("\n9.", "\n").replace("\n10.", "\n")
            text = text.strip().replace("\n\n", "\n").split("\n")
            text = [s.strip(' ,:-_;*()"') for s in text]
            for s in text:
                print(s)
            print()
            dataset[case + "_samples"] += text

    return dataset

#%% Generate datasets and save results

for rule_title in rules:
    dataset = generate_ds(rule_title)
        
    # Save generated dataset
    with open(OUTPUT_DIR + f"{rule_title}.json", "w") as file:
        json.dump(dataset, file)
# %%
