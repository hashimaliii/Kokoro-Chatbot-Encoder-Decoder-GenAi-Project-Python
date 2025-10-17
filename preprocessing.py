import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import math
import os
import sentencepiece as spm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

DATA_PERCENTAGE = 100

local_csv = 'emotion-emotion_69k.csv'

df = pd.read_csv(local_csv)

def normalize_text(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\s([?.!,;:])', r'\1', text)
    text = re.sub(r'([?.!,;:])(?=\w)', r'\1 ', text)
    return text

for col in ['Situation', 'emotion', 'empathetic_dialogues', 'labels']:
    df[col] = df[col].apply(normalize_text)

emotion_counts = df['emotion'].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 50].index
df = df[df['emotion'].isin(valid_emotions)]

if DATA_PERCENTAGE < 100:
    df = df.sample(frac=DATA_PERCENTAGE/100, random_state=42).reset_index(drop=True)
    print(f'Using {DATA_PERCENTAGE}% of data: {len(df)} samples')

from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

# Save the raw 80/10/10 splits to CSV files inside preprocessed/
OUT_DIR = 'preprocessed'
os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUT_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
print(f'Saved raw splits to {OUT_DIR}/train.csv, {OUT_DIR}/val.csv, {OUT_DIR}/test.csv')

train_text = train_df[['Situation', 'emotion', 'empathetic_dialogues', 'labels']].fillna('').agg(' '.join, axis=1)

# SentencePiece config
SP_MODEL_PREFIX = os.path.join(OUT_DIR, 'spm_emotion')
SP_MODEL_FILE = SP_MODEL_PREFIX + '.model'
SP_VOCAB_SIZE = 8000

def train_sentencepiece(corpus_iterable, model_prefix=SP_MODEL_PREFIX, vocab_size=SP_VOCAB_SIZE):
    # write temporary corpus file
    corpus_file = model_prefix + '_corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for line in corpus_iterable:
            f.write(line.replace('\n', ' ') + '\n')
    spm.SentencePieceTrainer.Train(
        f"--input={corpus_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --unk_id=3 --pad_id=0 --bos_id=1 --eos_id=2"
    )

# If a model is not present, train SentencePiece on the training text
if not os.path.exists(SP_MODEL_FILE):
    print('Training SentencePiece model...')
    # use the train_text Series as corpus; place corpus and model under OUT_DIR
    train_sentencepiece(train_text, model_prefix=SP_MODEL_PREFIX)

# Load trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_FILE)

# Build vocabulary mapping including special emotion tokens
SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>', '<sep>']
EMOTIONS = [f'<emotion_{e}>' for e in sorted(train_df['emotion'].unique())]

# We'll use SentencePiece ids for subwords; then reserve new ids for emotion tokens by extending
base_vocab_size = sp.get_piece_size()
vocab = {sp.id_to_piece(i): i for i in range(base_vocab_size)}
next_id = base_vocab_size
for tok in EMOTIONS:
    vocab[tok] = next_id
    next_id += 1

# Map SentencePiece special ids to our special names only if they exist and do not conflict
pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
unk_id = sp.unk_id()

# Build inverse vocab before possible additions
inv_vocab = {i: w for w, i in vocab.items()}

if pad_id != -1 and pad_id not in inv_vocab:
    vocab['<pad>'] = pad_id
    inv_vocab[pad_id] = '<pad>'
if bos_id != -1 and bos_id not in inv_vocab:
    vocab['<bos>'] = bos_id
    inv_vocab[bos_id] = '<bos>'
if eos_id != -1 and eos_id not in inv_vocab:
    vocab['<eos>'] = eos_id
    inv_vocab[eos_id] = '<eos>'
if unk_id != -1 and unk_id not in inv_vocab:
    vocab['<unk>'] = unk_id
    inv_vocab[unk_id] = '<unk>'

inv_vocab = {i: w for w, i in vocab.items()}

def encode(text, add_bos_eos=True):
    if text is None:
        text = ''
    pieces = sp.encode(text, out_type=str)
    ids = [vocab.get(p, vocab['<unk>']) for p in pieces]
    if add_bos_eos:
        ids = [vocab['<bos>']] + ids + [vocab['<eos>']]
    return ids

def decode(ids, remove_special=True):
    # map ids back to pieces; if id not in inv_vocab, use <unk>
    pieces = [inv_vocab.get(i, '<unk>') for i in ids]
    if remove_special:
        pieces = [p for p in pieces if p not in SPECIAL and not p.startswith('<emotion_')]
    # for pieces that are SentencePiece tokens, join using the processor
    # convert back to ids that SentencePiece knows (skip custom emotion tokens)
    sp_ids = [i for i in ids if i < base_vocab_size]
    text = sp.decode(sp_ids) if sp_ids else ''
    return text

print(f'Vocab size (including emotion tokens): {len(vocab)}')


# --- Build Input (X) and Target (Y) columns per user template ---
def build_input_text(emotion: str, situation: str, customer_utterance: str) -> str:
    """Format the model input (X).

    Example:
    Emotion: sentimental | Situation: I remember... | Customer: This was a best friend. I miss her.
    Agent:
    """
    emotion = emotion or ''
    situation = situation or ''
    customer_utterance = customer_utterance or ''
    # keep single spaces and trimmed
    return f"Emotion: {emotion} | Situation: {situation} | Customer: {customer_utterance}\nAgent:"


def build_target_text(agent_reply: str) -> str:
    """Format the model target (Y)."""
    return (agent_reply or '').strip()


# Apply to train/val/test splits and save processed CSVs
for split_df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
    # create input and target columns
    split_df['input_text'] = split_df.apply(
        lambda r: build_input_text(r.get('emotion'), r.get('Situation'), r.get('empathetic_dialogues')), axis=1
    )
    split_df['target_text'] = split_df['labels'].fillna('').astype(str).apply(build_target_text)

    out_csv = os.path.join(OUT_DIR, f"{name}_processed.csv")
    split_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(split_df)} rows")

    # optional: show a couple of examples
    print(f"Example {name} input/target:\n", split_df[['input_text', 'target_text']].head(2).to_dict(orient='records'))


# --- Create PyTorch Dataset and DataLoaders ---
class ChatDataset(Dataset):
    """Dataset that returns token id sequences for input_text and target_text."""
    def __init__(self, df, input_col='input_text', target_col='target_text', add_bos_eos=True):
        self.df = df.reset_index(drop=True)
        self.input_col = input_col
        self.target_col = target_col
        self.add_bos_eos = add_bos_eos

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = encode(row.get(self.input_col, ''), add_bos_eos=self.add_bos_eos)
        y = encode(row.get(self.target_col, ''), add_bos_eos=self.add_bos_eos)
        return {'input_ids': torch.tensor(x, dtype=torch.long), 'target_ids': torch.tensor(y, dtype=torch.long)}


def collate_fn(batch):
    """Pad batch of variable-length sequences and return tensors.

    Returns dict with keys: input_ids (B, T_in), input_lengths, target_ids (B, T_out), target_lengths
    """
    input_seqs = [item['input_ids'] for item in batch]
    target_seqs = [item['target_ids'] for item in batch]
    pad_id = vocab.get('<pad>', 0)
    # pad_sequence expects list of tensors (seq_len, ...) so set batch_first=True
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=pad_id)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=pad_id)
    input_lengths = torch.tensor([len(s) for s in input_seqs], dtype=torch.long)
    target_lengths = torch.tensor([len(s) for s in target_seqs], dtype=torch.long)
    return {
        'input_ids': input_padded,
        'input_lengths': input_lengths,
        'target_ids': target_padded,
        'target_lengths': target_lengths,
    }


BATCH_SIZE = 32
train_loader = DataLoader(ChatDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(ChatDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(ChatDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f'Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}')

