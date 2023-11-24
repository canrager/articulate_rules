# %%
import openai
import json

# %% Set Keys
with open("openaik.txt", "r") as file:
    openai.api_key = file.readline().strip()

# %% Set name of output files
rule = "german_only"
system_prompt = "You are a helpful assistant."
context_prompt = "We are labeling input strings as true or false using a simple rule. Here are four example inputs with the correct label: \n"
task_prompt = "\nPlease articulate the rule for correctly classifying the input strings in a single sentence using short and clear language."

examples_prompt = ""
with open(f"./outputs/{rule}.txt", "r") as file:
    dataset = json.load(file)
    for i in range(len(dataset["samples"])):
        inp = dataset["samples"][str(i)]["input"]
        lbl = dataset["samples"][str(i)]["label"]
        context_prompt += f'Input: {inp}, Label: {lbl}; '

print(context_prompt + examples_prompt + task_prompt)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": context_prompt + examples_prompt + task_prompt}
]


# %% OpenAI Text Generation
out = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
text = out.choices[0].message.content
print(text)

with open(f"./outputs/{rule}", "w") as file:
    json.dump(dataset, file)
# %%
