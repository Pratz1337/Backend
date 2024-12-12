import os
from os.path import expanduser
import requests
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from tqdm import tqdm

# Define model path
model_path = expanduser("~/Models/llama-2-7b-chat.Q4_0.gguf")
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf"  # Replace with actual URL

# Download model if not exists
def download_model(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Model not found. Downloading...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(path, "wb") as file, \
                tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Model") as pbar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                pbar.update(len(data))
        print("Download complete.")
    else:
        print("Model already exists.")

# Check and download the model
download_model(model_url, model_path)

# Load the Llama model
llm = LlamaCpp(
    model_path=model_path,
    streaming=False,
)

# Define LLM with memory and prompt
model = Llama2Chat(llm=llm)
prompt_template = PromptTemplate(input_variables=["text"], template="{text}")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

# Example query
response = chain.run(
    text="What can I see in Vienna? Propose a few locations. Names only, no details."
)
print(response)
