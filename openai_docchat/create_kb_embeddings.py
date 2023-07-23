import os
from PyPDF2 import PdfReader
import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key using the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the current working directory and construct the full path to /kb_docs folder
project_dir = os.getcwd()
pdfs_dir = os.path.join(project_dir, "kb_docs")

# Initialize an empty DataFrame to store the text and embeddings
embeddings_df = pd.DataFrame(columns=['text', 'embedding'])

for filename in os.listdir(pdfs_dir):
    if filename.endswith(".pdf"):
        full_path = os.path.join(pdfs_dir, filename)  # Construct the full path to the file
        reader = PdfReader(full_path)
        full_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text
        embeddings_df = embeddings_df.append({'text': full_text, 'embedding': ""}, ignore_index=True)

# Generate embeddings for each text using OpenAI's get_embedding() function
embeddings_df['embedding'] = embeddings_df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

# Save the DataFrame to a CSV file
embeddings_df.to_csv('kb_doc_embeddings_df.csv')
