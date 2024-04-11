import pandas as pd
import tiktoken
from openai import OpenAI
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = OpenAI(max_retries=5)

embedding_model = "text-embedding-3-large"
embedding_encoding = "cl100k_base"
max_tokens = 8000 

encoding = tiktoken.get_encoding(embedding_encoding)

client = OpenAI()

def split_text(text):
    '''
    Split the text into chunks based on token limit.
    '''
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8050, 
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
    try:
        texts = text_splitter.split_text(text)
        return texts
    except Exception as e:
        print(f"An error occurred while splitting text: {str(e)}")
        return [text]

def reduce_df(df):
    '''
    This method ensures that the request sent to OpenAI API will not exceed its limit.
    If a row exceeds the token limit, it splits the text into chunks and creates new rows.
    '''
    try:
        encoding = tiktoken.get_encoding(embedding_encoding)
        df["n_tokens"] = [len(encoding.encode(x)) for x in df["Text"]]
        
        new_rows = []
        for index, row in df.iterrows():
            if row["n_tokens"] > max_tokens:
                texts = split_text(row["Text"])
                for text in texts:
                    new_row = row.copy()
                    new_row["Text"] = text
                    new_rows.append(new_row)
            else:
                new_rows.append(row)
        
        new_df = pd.DataFrame(new_rows)
        new_df.drop(columns=["n_tokens"], inplace=True)

        return new_df
    except Exception as e:
        print(f"An error occurred while reducing DataFrame: {str(e)}")
        return None

def get_embedding(text, model="text-embedding-3-large"):
   #TODO: change the model
   '''
   This method create embedding for a single string, using OPENAI API.
   '''
   try:
       text = text.replace("\n", " ")
       emb = client.embeddings.create(input=[text], model=model).data[0].embedding
       return emb
   except Exception as e:
       print(f"An error occurred while creating embedding: {str(e)}")
       return None