import openai

import json
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm, trange
rng = default_rng()

class ArticulationExperiment():
    def __init__(
        self,
        model,
        dataset_dir,
        n_exp,
        n_fewshot_examples,
        n_tasks,
        system_prompt,
        context_prompt,
        articulation_prompt,
        classification_prompt,
        multiple_classifications_prompt
        ) -> None:

        self.model = model
        self.dataset_dir = dataset_dir
        self.n_exp = n_exp
        self.n_fewshot_examples = n_fewshot_examples
        self.n_tasks = n_tasks
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt
        self.articulation_prompt = articulation_prompt
        self.classification_prompt = classification_prompt
        self.multiple_classifications_prompt = multiple_classifications_prompt

    def get_fewshot_examples(self, rule):
        '''
        Organizes random samples from the dataset into a few-shot prompt.

        rule: name of the rule (and dataset file)
        n_examples: number of few-shot examples per true/false class in prompt
        '''
        with open(self.dataset_dir + f"{rule}.json", "r") as file:
            dataset = json.load(file)

        n_true = len(dataset["true_samples"])
        n_false = len(dataset["false_samples"])

        # Draw random samples from datset
        true_sample_idxs = rng.choice(n_true, size=self.n_fewshot_examples+1, replace=False) # The +1 dimesion holds the sample which will be prompted to label
        false_sample_idxs = rng.choice(n_false, size=self.n_fewshot_examples+1, replace=False)
        true_samples = np.array(dataset['true_samples'])[true_sample_idxs[:-1]]
        false_samples = np.array(dataset['false_samples'])[false_sample_idxs[:-1]]
        true_labeled = [[ts, "True"] for ts in true_samples]
        false_labeled = [[ts, "False"] for ts in false_samples]
        labeled_examples = np.vstack((true_labeled, false_labeled))
        #rng.shuffle(labeled_examples)

        # Put samples into a single string
        examples_prompt = ""
        for ex, lbl in labeled_examples:
            examples_prompt += f'Input: {ex} Label: {lbl}\n'

        # Choose a true or false prompt
        if rng.choice([True, False]):
            task_input = dataset['true_samples'][true_sample_idxs[-1]]
            task_input = f'Input: {task_input} Label:'
            task_label = "True"
        else:
            task_input = dataset['false_samples'][false_sample_idxs[-1]]
            task_input = f'Input: {task_input} Label:'
            task_label = "False"
        return examples_prompt, task_input, task_label
    
    def get_examples_and_tasks(self, rule, shuffle_examples=True, shuffle_tasks=True):
        '''
        Organizes random samples from the dataset into a few-shot prompt.

        rule: name of the rule (and dataset file)
        n_examples: number of few-shot examples per true/false class in prompt
        n_tasks: number of input sentences the model has to label.
        '''
        # Load dataset
        with open(self.dataset_dir + f"{rule}.json", "r") as file:
            dataset = json.load(file)

        n_true = len(dataset["true_samples"])
        n_false = len(dataset["false_samples"])

        # Draw random samples from datset
        true_sample_idxs = rng.choice(n_true, size=self.n_fewshot_examples+self.n_tasks, replace=False) # The +1 dimesion holds the sample which will be prompted to label
        true_labeled = np.array([[ts, "True"] for ts in dataset['true_samples']])
        true_examples = true_labeled[true_sample_idxs[:self.n_fewshot_examples]]
        true_tasks = true_labeled[true_sample_idxs[self.n_fewshot_examples:]]

        false_sample_idxs = rng.choice(n_false, size=self.n_fewshot_examples+self.n_tasks, replace=False)
        false_labeled = np.array([[ts, "False"] for ts in dataset['false_samples']])
        false_examples = false_labeled[false_sample_idxs[:self.n_fewshot_examples]]
        false_tasks = false_labeled[false_sample_idxs[self.n_fewshot_examples:]]

        # Put examples into prompt
        labeled_examples = np.vstack((true_examples, false_examples))
        if shuffle_examples:
            rng.shuffle(labeled_examples)

        examples_prompt = ""
        for ex, lbl in labeled_examples:
            examples_prompt += f'Input: {ex} Label: {lbl}\n'
        
        # Put tasks into prompt
        labeled_tasks = np.vstack((true_tasks, false_tasks))
        if shuffle_tasks:
            rng.shuffle(labeled_tasks)

        task_inputs = ""
        task_labels = []
        for i, labeled_task in enumerate(labeled_tasks):
            ts, lbl = labeled_task
            task_inputs += f"({i}) {ts}\n"
            task_labels.append(lbl)

        return examples_prompt, task_inputs, task_labels

    def generate_rule_articulation(self, fewshot_examples):
        messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.context_prompt + fewshot_examples + self.articulation_prompt}
        ]
        # OpenAI Text Generation
        out = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
        )
        text = out.choices[0].message.content
        return text
    
    def do_classification_experiment(self, rule, verbose):
        labels = np.zeros(self.n_exp)
        preds = np.zeros(self.n_exp)

        for i in tqdm(range(self.n_exp),):
            fewshot_examples, task_input, label = self.get_fewshot_examples(rule)

            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.context_prompt + fewshot_examples + self.classification_prompt + task_input}
            ]
            # OpenAI Text Generation
            out = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )
            prediction = out.choices[0].message.content
            prediction.strip(' ,./123567890;:"')

            if ("True" in prediction) and not ("False" in prediction):
                preds[i] = 1
                if label == "True":
                    labels[i] = 1
            elif ("False" in prediction) and not ("True" in prediction):
                preds[i] = 0
                if label == "True":
                    labels[i] = 1
            else:
                print(f"\n Input cannot be converted into prediction: {prediction}")

            if verbose:
                print(self.context_prompt + fewshot_examples + self.classification_prompt + task_input)
                print(f'{prediction=}')
                print(f'{label=}\n\n')

        if verbose:
            print(f'{labels=}')
            print(f'{preds=}')
        
        return labels, preds
    
    def classify(self, rule, verbose=False):
        accs = np.zeros(self.n_exp)
        for j in trange(self.n_exp):
            fewshot_examples, task_inputs, task_labels = self.get_examples_and_tasks(rule=rule)
            msg = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.context_prompt + fewshot_examples + self.multiple_classifications_prompt + task_inputs}
            ]
            # OpenAI Text Generation
            out = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=msg,
            )
            predictions = out.choices[0].message.content
            predictions = predictions.split('\n')
            if verbose:
                print(predictions)

            if len(predictions) != len(task_labels):
                print(f"Evaluation of prompt #{j} failed: predictions can't be parsed automatically")
                continue

            if verbose:
                for i in range(2*self.n_tasks):
                    print(f'{predictions[i]} ({task_labels[i]} is correct)')

            predictions = [p.strip(' ,./1234567890;:"()') for p in predictions]
            if len(predictions) != len(task_labels):
                print(f"Evaluation of prompt #{j} failed: predictions can't be parsed automatically")
                continue

            preds = np.zeros(2*self.n_tasks)
            labels = np.zeros(2*self.n_tasks)
            for i in range(2*self.n_tasks):
                if ("True" in predictions[i]) and not ("False" in predictions[i]):
                    preds[i] = 1
                    if task_labels[i] == "True":
                        labels[i] = 1
                elif ("False" in predictions[i]) and not ("True" in predictions[i]):
                    preds[i] = 0
                    if task_labels[i] == "True":
                        labels[i] = 1
                else:
                    if verbose:
                        print(f"\n Input cannot be converted into prediction: {predictions[i]}")
                    
            accs[j] = np.sum(preds == labels) / len(preds)
        return accs