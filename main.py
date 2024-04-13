import gradio as gr
import openai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


# Define the Python function that takes three inputs and returns a result

llm_models = {"GPT-3.5": "gpt-3.5-turbo-0125", "GPT-4": "gpt-4-turbo"}

INDEX_NAME = "mtp"

# Initialize Pinecone client
pc = Pinecone(api_key='47703291-8727-4f0c-a780-571542ece3a3')

index = pc.Index(
    host='https://mtp-9iwsr1u.svc.aped-4627-b74a.pinecone.io'
    )

def RAG(prompt_text):
    model = SentenceTransformer("all-mpnet-base-v2")

    embeddings = model.encode(prompt_text)
    context = gradio_interface(embeddings)

    query = perform_rag(prompt_text,context)


def perform_rag(context, top_context):
  # Load your chosen RAG model and tokenizer (replace placeholders)
  model = SentenceTransformer("all-mpnet-base-v2")

  tokenizer = AutoTokenizer.from_pretrained(model)
#   model = AutoModelForSeq2SeqLM.from_pretrained(model)

  # Prepare RAG inputs (replace with your processing steps)
  input_ids = tokenizer(context, top_context, return_tensors="pt")["input_ids"]

  # Generate text using your RAG model (replace with your logic)
  output = model.generate(**input_ids)
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

  return generated_text


def gradio_interface(context):
    print("Context Is :")
    print()
    top_k = 2
    # Perform retrieval using Pinecone based on the context
    vec = context.tolist()

    print(type(vec))
    print(vec)

    print("Embeddings are ")
    embeddings = index.query(
    vector=vec,
    top_k=top_k,
    include_values=True)

    print(embeddings)
    all_values = [match['values'] for match in embeddings['matches']]

  # List to store top contexts and their generated text
    top_contexts_and_responses = []

    # Loop through top k results and perform RAG for each
    for top_context in all_values:
        print(top_context)
        # top_context = response["document"]
        # Replace with your RAG implementation
        generated_text = perform_rag(vec, top_context)  # Assuming embedding field name
        print("generated text is",  generated_text)
        # top_contexts_and_responses.append((top_context, generated_text))

    # Return list of top contexts and generated responses
    return top_contexts_and_responses


def process_text(input1, input2, input3, key, model_in, temp_in):
    # openai.api_key = "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"
    curr_string = "You are an oncologist. If for a clinical trial, the Inclusion Criteria is :\n" + \
        input2 + "\n" + \
        "and the Exclusion Criteria is : " + input3 + "\n" + \
        "and the Patient Profile is :" + \
        input1 + \
        ".\n Then given this information determine if the patient is eligible for the trial. Give a binary answer Eligible/Not Eligible as the response. In the response please do not add any extra text, just the response Eligible or Not Eligible"

    # TO perform RAG
    query = RAG(curr_string)
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
