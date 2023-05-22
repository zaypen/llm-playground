import gradio as gr
from models import models
from prompts import prompts


def on_model_change(name):
    try:
        llm = models.get(name)()
    except RuntimeError as e:
        return None, [(e, 'ERROR')]
    return llm, [(name, 'OK')]


def on_generate(llm, prompt):
    if llm:
        return llm, llm(prompt)
    return llm, '<ERROR>'


def on_prompt_change(name):
    try:
        return prompts.get(name)
    except RuntimeError as e:
        print(e)
        return ''


model_names = list(models.keys())
prompt_names = list(prompts.keys())

with gr.Blocks() as app:
    model = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(label='Model', choices=model_names, value=model_names[0])
            model_status = gr.HighlightedText(label='Status').style(color_map={'OK': 'green', 'ERROR': 'red'})
        with gr.Column(scale=3):
            prompt_name = gr.Dropdown(label='Preset', choices=prompt_names, value=prompt_names[0])
            input_textbox = gr.Textbox(label='Input', lines=5)
            process = gr.Button()
            model_output = gr.Textbox(label='Output', lines=5)

    model_name.change(on_model_change, [model_name], [model, model_status])
    prompt_name.change(on_prompt_change, [prompt_name], [input_textbox])
    process.click(on_generate, [model, input_textbox], [model, model_output])

if __name__ == "__main__":
    app.launch()
