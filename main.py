import gradio as gr
import openai

# Define the Python function that takes three inputs and returns a result

llm_models = {"GPT-3.5": "gpt-3.5-turbo-0125", "GPT-4": "gpt-4-turbo"}


def process_text(input1, input2, input3, key, model_in, temp_in):
    openai.api_key = key
    curr_string = "You are an oncologist. If for a clinical trial, the Inclusion Criteria is :\n" + \
        input2 + "\n" + \
        "and the Exclusion Criteria is : " + input3 + "\n" + \
        "and the Patient Profile is :" + \
        input1 + \
        ".\n Then given this information determine if the patient is eligible for the trial. Give a binary answer Eligible/Not Eligible as the response. In the response please do not add any extra text, just the response Eligible or Not Eligible"

    prompt_parts = curr_string
    print(prompt_parts)
    response = openai.ChatCompletion.create(
        model=llm_models[model_in],
        messages=[{"role": "system", "content": "You are a helpful assistant and acting as an oncologist"}, {
            "role": "user", "content": prompt_parts}],
        temperature=temp_in
    )
    print(response)
    return (response["choices"][0]["message"]["content"])


# Create input boxes for the three inputs
input_box1 = gr.Textbox(label="Patient Profile")
input_box2 = gr.Textbox(label="Inclusion Criteria")
input_box3 = gr.Textbox(label="Exclusion Criteria")
input_box4 = gr.Textbox(label="API Key")

output_box = gr.Textbox(label="Verdict")
# Create the Gradio interface
image_path = "logo.png"
with gr.Blocks() as demo:
    # gr.Image("logo.png", height="20px", show_label=False,
    #          show_download_button=False, show_share_button=False)
    big_block = gr.HTML("""
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Syneos_Health_logo.svg/1200px-Syneos_Health_logo.svg.png" alt="logo" width="150" height="150">
    """)
    gr.Interface(fn=process_text,
                 inputs=[input_box1, input_box2, input_box3, input_box4, gr.Dropdown(
                        ["GPT-3.5", "GPT-4"], label="LLM Model", info="Select a LLM model"), gr.Slider(0, 1, value=0.9, label="Temperature", info="Select a Temperature"),
                 ],
                 outputs=output_box,
                 title="Predict Patient Eligibility",
                 description="Enter Patient Profile and Inclusion and Exclusion criteria to check Eligibility of patient",
                 allow_flagging="never",
                 )
demo.launch()
