# %%
import openai

# %% Set name of output files
run_name = "pirates_journey"

# %% Set Keys
with open("openaik.txt", "r") as file:
    openai.api_key = file.readline().strip()

# %% OpenAI Text Generation
out = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a foolish captain of crew of pirates who just returned from a wild journey.",
        },
        {"role": "user", "content": "What did you encounter on your journey?"},
    ],
)
text = out.choices[0].message.content
print(text)

with open(f"./outputs/{run_name}", "a") as file:
    file.write(text)

# %%
