from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms import OpenAI, Cohere, HuggingFacePipeline

load_dotenv()

pipeline_kwargs = {
    "max_new_tokens": 256
}

models = {
    'GPT-3 (OpenAI)': lambda: OpenAI(),
    'Cohere GPT (Cohere)': lambda: Cohere(),
    'GPT-2 124M (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'gpt2', **pipeline_kwargs)
    ),
    'GPT-2 355M (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'gpt2-medium', **pipeline_kwargs)
    ),
    'GPT-2 774M (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'gpt2-large', **pipeline_kwargs)
    ),
    'Flan T5 Large (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text2text-generation', "google/flan-t5-large", **pipeline_kwargs)
    ),
    'BLOOMZ 650M (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'bigscience/bloomz-560m', **pipeline_kwargs)
    ),
    'BLOOMZ 1B1 (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'bigscience/bloomz-1b1', **pipeline_kwargs)
    ),
    'BLOOMZ 1B7 (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'bigscience/bloomz-1b7', **pipeline_kwargs)
    ),
    'OPT 1B3 (Local)': lambda: HuggingFacePipeline(
        pipeline=pipeline('text-generation', 'facebook/opt-1.3b', **pipeline_kwargs)
    ),
}
