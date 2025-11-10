'''
Code to test inspect_ai package from AI Security Institute.

Adapted from code:
https://inspect.aisi.org.uk/#sec-hello-inspect

Installation:
    python3.10 -m venv inspect_ai_venv
    source inspect_ai_venv/bin/activate  # On Windows use `inspect_ai_venv\Scripts\activate`
    pip install -r requirements_ai_security_evals.txt
    pip install inspect-ai
    pip instal openai

Usage:
    inspect eval ai_security_evals.py --model openai/gpt-4

Author: Soumya Banerjee

Date: November 2025

'''

import inspect_ai
from inspect_ai import task, Task
from inspect_ai.dataset import example_dataset
from inspect_ai.scorer import model_graded_fact 
from inspect_ai.solver import chain_of_thought, generate, self_critique
#from .tasks import theory_of_mind 
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# define the task
@task
def theory_of_mind():
    '''
    Theory of Mind evaluation task.
    '''

    return Task(
        dataset = example_dataset("theory_of_mind"), # using example dataset
        solver = [chain_of_thought(), generate(), self_critique()], # using multiple solvers
        scorer = model_graded_fact(), # using model graded factual scorer since the output will be natural language
    )



# The @task decorator registers the function as a task within the inspect_ai framework.
# The function theory_of_mind returns a Task object that encapsulates the dataset, solvers,
# and scorer to be used for the evaluation.

# The @task decorator applied to the theory_of_mind() function is what enables inspect eval to find and run the eval in the source file passed to it. For example, here we run the eval against GPT-4:

# call this using the command line:
# inspect eval ai_security_evals.py --model openai/gpt-4