import pandas as pd
import gradio as gr
import openai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

llm_models = {"GPT-3.5": "gpt-3.5-turbo-0125", "GPT-4": "gpt-4-turbo"}

INDEX_NAME = "mtp"

pc = Pinecone(api_key='47703291-8727-4f0c-a780-571542ece3a3')

index = pc.Index(
    host='https://mtp-9iwsr1u.svc.aped-4627-b74a.pinecone.io'
)


def RAG(prompt_text):
    model = SentenceTransformer("all-mpnet-base-v2")

    embeddings = model.encode(prompt_text)
    context = gradio_interface(embeddings)
    return context


df = pd.read_csv("Data.csv", encoding='cp1252')


def gradio_interface(context):

    top_k = 2
    vec = context.tolist()
    embeddings = index.query(
        vector=vec,
        top_k=top_k,
        include_values=True)
    all_values = [match['id'] for match in embeddings['matches']]
    ret = "\nHere is the first set of criteria, patient profile and result on eligibility:"
    for j in range(top_k):
        print("Id is ", all_values[j])

        i = int(all_values[j])
        r = ""
        r = "\n Inclusion Criteria for the trial is "
        r += "\n"
        r += df['ï»¿inclusion_criteria'][i]

        r += "\n Exclusion Criteria for the trial is "
        r += "\n"
        r += df['exclusion_criteria'][i]

        r += "\n Patient Profile is "
        r += "\n"
        r += df['patient_profiles'][i]

        r += "\n The eligibility of the patient for the clinical trial is: "
        r += "\n"
        r += df['target'][i]

        if j != top_k-1:
            r += " \n Here is another patient profile: "

        ret += r
    return ret


def process_text(input1, input2, input3, key, model_in, temp_in, rag_check):
    print("RAG : ", rag_check)
    openai.api_key = key
    curr_string = "You are an oncologist. If for a clinical trial, the Inclusion Criteria is :\n" + \
        input2 + "\n" + \
        "and the Exclusion Criteria is : " + input3 + "\n" + \
        "and the Patient Profile is :" + \
        input1 + \
        ".\n Then given this information determine if the patient is eligible for the trial. Give a binary answer Eligible/Not Eligible as the response. In the response please do not add any extra text, just the response Eligible or Not Eligible. After giving the verdict in one line, from the next line give the reasons for giving the verdict as eligible or not eligible"
    no_rag_case = curr_string
    curr_string += "\nTo determine the eligibility, you may consider as an example, some similar patients with inclusion criteria, exclusion criteria, and a verdict on its eligibility below to help you better make your decision about the eligibility:\n"

    query = RAG(curr_string)
    curr_string += query

    curr_string = curr_string if rag_check else no_rag_case
    print(curr_string)

    response = openai.ChatCompletion.create(
        model=llm_models[model_in],
        messages=[{"role": "system", "content": "You are a helpful assistant and acting as an oncologist"}, {
            "role": "user", "content": curr_string}],
        temperature=temp_in
    )
    print(response)
    llm_reasoning = ""
    for point in response["choices"][0]["message"]["content"].split("\n")[1:]:
        llm_reasoning += point + "\n"
    return (response["choices"][0]["message"]["content"].split("\n")[0], llm_reasoning)


input_box1 = gr.Textbox(label="Patient Profile")
input_box2 = gr.Textbox(label="Inclusion Criteria")
input_box3 = gr.Textbox(label="Exclusion Criteria")
input_box4 = gr.Textbox(label="API Key")

output_box1 = gr.Textbox(label="Verdict")
output_box2 = gr.Textbox(label="LLM Reason")

image_path = "logo.png"
with gr.Blocks() as demo:
    big_block = gr.HTML("""
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Syneos_Health_logo.svg/1200px-Syneos_Health_logo.svg.png" alt="logo" width="150" height="150">
    """)
    gr.Interface(fn=process_text,
                 inputs=[input_box1, input_box2, input_box3, input_box4, gr.Dropdown(
                        ["GPT-3.5", "GPT-4"], label="LLM Model", info="Select a LLM model"), gr.Slider(0, 1, value=0.9, label="Temperature", info="Select a Temperature"),
                     gr.Checkbox(label="Apply RAG",
                                 info="Check the following to use RAG", value=True),
                 ],
                 outputs=[output_box1, output_box2],
                 title="Predict Patient Eligibility",
                 description="Enter Patient Profile and Inclusion and Exclusion criteria to check Eligibility of patient",
                 allow_flagging="never",
                 )
demo.launch()
