
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    # Tokenize the text and get token ids
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move input tensors to GPU
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get BERT model outputs (hidden states)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        
    # We will use the embeddings from the [CLS] token (the first token in the input)
    embeddings = outputs.last_hidden_state[0, 0, :].cpu().numpy()  # Move back to CPU for further processing
    return embeddings

# Apply BERT embedding extraction with progress bar
embeddings = []
for text in tqdm(df['processed_content'], desc="Extracting BERT embeddings", unit="text"):
    embeddings.append(get_bert_embeddings(text))

# Store the embeddings in the dataframe
df['bert_embeddings'] = embeddings

# Check the shape of the embeddings
print(f"BERT Embedding Shape: {df['bert_embeddings'][0].shape}")




import fasttext
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load pre-trained FastText model (this is the FastText English model)
ft = fasttext.load_model('cc.en.300.bin')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # Get embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[0, 0, :].cpu().numpy()  
    return embeddings

# Function to get FastText embeddings
def get_fasttext_embeddings(text):
    # FastText averages the word embeddings in the text
    words = text.split()
    embeddings = np.zeros(300)  # FastText embedding size is 300
    count = 0
    for word in words:
        embeddings += ft.get_word_vector(word)
        count += 1
    if count > 0:
        embeddings /= count  # Average the word embeddings
    return embeddings

# Apply BERT and FastText embedding extraction with progress bar
final_embeddings = []
for text in tqdm(df['processed_content'], desc="Extracting embeddings", unit="text"):
    bert_emb = get_bert_embeddings(text)
    fasttext_emb = get_fasttext_embeddings(text)
    
    # Concatenate the embeddings
    combined_emb = np.concatenate((bert_emb, fasttext_emb))  # 768 (BERT) + 300 (FastText) = 1068
    final_embeddings.append(combined_emb)

# Store the concatenated embeddings in the dataframe
df['combined_embeddings'] = final_embeddings

# Check the shape of the concatenated embeddings
print(f"Concatenated Embedding Shape: {df['combined_embeddings'][0].shape}")

