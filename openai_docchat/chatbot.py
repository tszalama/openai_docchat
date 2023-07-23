import os
import openai
import gradio as gr
from openai.embeddings_utils import cosine_similarity, get_embedding
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv

# Function to convert a string representation of an array to a NumPy array
def string_to_array(s):
    return np.fromstring(s.strip('[]'), sep=',')

# Method for generating responses using regular models (non-gpt-3.5-turbo)
def get_regular_response(context, question):
    global model_selection
    global mode_selection
    model = ""
    if model_selection == "CoQA curie":
        model = "curie:ft-personal:coqa-curie-v2-2023-04-18-20-20-08"
    elif model_selection == "text-davinci-003":
        model = "text-davinci-003"
    else:
        print("[ERROR] matching model not found")
        return "", ""
    formatted_prompt = f"Answer the following question using only the given Context. If you don't know the answer for certain, say I don't know.\n\nContext:\n{context}\n" + f"\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        prompt=formatted_prompt,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=model
    )
    return response["choices"][0]["text"], formatted_prompt, int(response['usage']['total_tokens'])

# Method for generating responses using gpt-3.5-turbo model
def get_gpt_3_5_response(context, question):
    formatted_prompt = f"Answer the following question using only the given Context. If you don't know the answer for certain, say I don't know.\n\nContext:\n{context}\n" + f"\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=1,
        max_tokens=200,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt},
        ]
    )
    return response['choices'][0]['message']['content'], formatted_prompt, int(response['usage']['total_tokens'])

# Method for getting the standalone question from a conversation history and input question
def get_query(question, history):
    conversation_history = ""
    for qa_pair in history:
        conversation_history = conversation_history + f"\nQuestion: {qa_pair[0]}" + f"\nAnswer: {qa_pair[1]}"
    # The following text template is taken from https://github.com/mayooear/gpt4-pdf-chatbot-langchain/blob/main/utils/makechain.ts
    prompt = f"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\nChat History:\n{conversation_history}\nFollow Up Input: {question}\nStandalone question:"
    global model_selection
    response = None
    response_text = ""
    if model_selection == "gpt-3.5":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            top_p=1,
            max_tokens=200,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        response_text = response['choices'][0]['message']['content']
    else:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_text = response["choices"][0]["text"]
    print("----------------------------")
    print("get_query\n\n")
    print(prompt)
    print(response_text)
    print("----------------------------")
    return response_text, int(response['usage']['total_tokens'])

# Method for searching a document using cosine similarity
def document_search(question):
    question_vector = get_embedding(question, engine="text-embedding-ada-002")
    embeddings_df["similarities"] = embeddings_df['embedding'].apply(lambda x: cosine_similarity(x, question_vector))
    similar_texts = embeddings_df.sort_values("similarities", ascending=False)
    context = similar_texts["text"].iloc[0] + similar_texts["text"].iloc[1]
    return context

# Method for coordinating document search, response generation, and result logging
def answer_question(question, history):
    start = time.perf_counter()
    standalone_question, query_tokens_used = get_query(question, history)
    context = document_search(standalone_question)

    global model_selection
    answer_tokens_used = 0
    if model_selection == "gpt-3.5":
        response, formatted_prompt, answer_tokens_used = get_gpt_3_5_response(context, standalone_question)
    else:
        response, formatted_prompt, answer_tokens_used = get_regular_response(context, standalone_question)
    request_time = time.perf_counter() - start

    print("----------------------------")
    print("Response from model:" + model_selection + "\n\n")
    print(formatted_prompt + response)
    print("----------------------------")

    global model_answer_log_df
    if os.path.exists("model_answer_log_df.csv"):
        model_answer_log_df = pd.read_csv('model_answer_log_df.csv')
    model_answer_log_df = model_answer_log_df.append({'prompt': formatted_prompt, 'question': question, 'answer': response, 'tokens_used': (query_tokens_used + answer_tokens_used), 'request_time': request_time}, ignore_index=True)
    model_answer_log_df.to_csv('model_answer_log_df.csv')
    return response

# Method for handling a new user question event
def document_chatbot(input, history):
    history = history or []
    output = answer_question(input, history)
    history.append((input, output))
    return history, history

# Method for updating the model selection
def update_model_selection(input, history):
    global model_selection
    model_selection = input
    print("----------------------------")
    print("Model changed to " + model_selection)
    print("----------------------------")

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read document embeddings DataFrame
embeddings_df = pd.read_csv('kb_doc_embeddings_df.csv')
embeddings_df['embedding'] = embeddings_df['embedding'].apply(string_to_array)

# Initialize DataFrame for logging model answers
model_answer_log = {'prompt': [], 'question': [], 'answer': [], 'tokens_used': [], 'request_time': []}
model_answer_log_df = pd.DataFrame(model_answer_log)

# Initialize default model and mode selections
model_selection = "text-davinci-003"
mode_selection = "zero-shot"

block = gr.Blocks()

# Gradio interface definition
with block:
    gr.Markdown("""<h1><center>Question Answering System</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Enter a question")
    state = gr.State()
    submit = gr.Button("SEND")
    radio = gr.Radio(["CoQA curie", "text-davinci-003", "gpt-3.5"], label="Model", value="text-davinci-003")
    radio.change(update_model_selection, inputs=[radio, state])
    submit.click(document_chatbot, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=True)