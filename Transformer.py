# Transformer model from GPT

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def encode(text):
    return torch.tensor([word2idx.get(t, unk_idx) for t in tokenize(text)])

class DataFramePairDataset(Dataset):
    def __init__(self, df):
        self.text1 = df["text1"].tolist()
        self.text2 = df["text2"].tolist()
        self.labels = df["label"].tolist()

    def __getitem__(self, idx):
        x1 = encode(self.text1[idx])
        x2 = encode(self.text2[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return x1, x2, label

    def __len__(self):
        return len(self.text1)

def collate_fn(batch):
    x1, x2, y = zip(*batch)
    x1 = pad_sequence(x1, batch_first=True, padding_value=pad_idx)
    x2 = pad_sequence(x2, batch_first=True, padding_value=pad_idx)
    y = torch.stack(y)
    return x1, x2, y

dataset = DataFramePairDataset(df)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)



















import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # non-trainable

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerSiameseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, nhead=4, ff_dim=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_len=100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def encode(self, x):
        emb = self.embedding(x)  # (1, seq_len, embed_dim)
        emb = self.positional_encoding(emb)
        encoded = self.encoder(emb)
        return encoded.mean(dim=1)  # mean-pooling

    def forward(self, x1, x2):
        v1 = self.encode(x1)
        v2 = self.encode(x2)
        return v1, v2


import torch.nn.functional as F

def cosine_bce_loss(v1, v2, label):
    sim = F.cosine_similarity(v1, v2)  # → shape: (batch_size,)
    loss = F.binary_cross_entropy((sim + 1) / 2, label)  # normalize sim to [0,1]
    return loss

def cosine_contrastive_loss(v1, v2, label, margin=0.5):
    sim = F.cosine_similarity(v1, v2)
    pos_loss = (1 - sim) * label
    neg_loss = torch.clamp(sim - margin, min=0.0) * (1 - label)
    return (pos_loss + neg_loss).mean()

def contrastive_loss(v1, v2, label, margin=1.0):
    dist = torch.norm(v1 - v2, p=2, dim=1)
    loss = label * dist**2 + (1 - label) * torch.clamp(margin - dist, min=0)**2
    return loss.mean()

from torch.nn.utils.rnn import pad_sequence

def tokenize(text): return text.lower().split()

# Example dataset
pairs = [
    ("nike red shoes", "red nike sneakers", 1),
    ("blue jeans", "smartphone case", 0),
    ("iphone charger", "apple charger", 1),
]

# Build vocab
from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(tokenize(x) for pair in pairs for x in pair[:2], specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])
pad_idx = vocab["<pad>"]

def encode(text): return torch.tensor([vocab[token] for token in tokenize(text)])

encoded_pairs = []
for a, b, label in pairs:
    encoded_pairs.append((encode(a), encode(b), torch.tensor(label, dtype=torch.float)))

from gensim.models import Word2Vec

# Sample tokenized corpus
sentences = [
    ["this", "is", "an", "example", "sentence"],
    ["word2vec", "in", "gensim", "is", "powerful"],
    ["we", "are", "training", "word", "embeddings"],
    ["word2vec", "works", "on", "sequences", "of", "words"],
]

# Train Word2Vec model
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # embedding dimension
    window=5,             # context window size
    min_count=1,          # minimum word frequency
    workers=4,            # use 4 threads
    sg=1,                 # 1 = skip-gram, 0 = CBOW
    epochs=10             # number of training iterations
)

# Save model (optional)
model.save("word2vec.model")

# Example: Get vector for a word
print("Vector for 'word2vec':")
print(model.wv["word2vec"])

# Most similar words
print("\nWords similar to 'word2vec':")
print(model.wv.most_similar("word2vec"))

