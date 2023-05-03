import os
from PyPDF2 import PdfReader
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding

openai.api_key = #INSERT YOUR OPENAI API KEY HERE

embeddings_df_structure = {'text' : [],'embedding': []}
embeddings_df = pd.DataFrame(embeddings_df_structure)

pdfs_dir = #Insert link to directory that contains your PDFs

for filename in os.listdir(pdfs_dir):
    if filename.endswith(".pdf"):
        full_path = os.path.join(pdfs_dir, filename)  # Construct the full path to the file
        reader = PdfReader(full_path)
        full_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += text
        embeddings_df = embeddings_df.append({'text' : full_text,'embedding': ""}, ignore_index=True)

embeddings_df['embedding'] = embeddings_df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

embeddings_df.to_csv('doc_embeddings_df.csv')