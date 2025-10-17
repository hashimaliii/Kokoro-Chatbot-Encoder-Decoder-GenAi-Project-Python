import os
import json
import math
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import argparse
import csv

# -------------------------
# Configuration / Hyperparams
# -------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 1337
torch.manual_seed(SEED)

# Model / training hyperparameters
EMBED_DIM = 256
NHEADS = 2
ENC_LAYERS = 2
DEC_LAYERS = 2
FFN_DIM = EMBED_DIM * 4
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
BETAS = (0.9, 0.98)

MAX_TARGET_LEN = 128

SP_MODEL_FILE = 'spm_emotion.model'
OUT_DIR = 'preprocessed'
SP_MODEL_FILE = os.path.join(OUT_DIR, 'spm_emotion.model') if os.path.exists(os.path.join(OUT_DIR, 'spm_emotion.model')) else 'spm_emotion.model'

def _choose_csv(name):
    # Prefer processed in OUT_DIR, then processed in cwd, then raw in OUT_DIR, then raw in cwd
    paths = [os.path.join(OUT_DIR, f"{name}_processed.csv"), f"{name}_processed.csv", os.path.join(OUT_DIR, f"{name}.csv"), f"{name}.csv"]
    for p in paths:
        if os.path.exists(p):
            return p
    return f"{name}.csv"

TRAIN_CSV = _choose_csv('train')
VAL_CSV = _choose_csv('val')
TEST_CSV = _choose_csv('test')

CHECKPOINT_DIR = 'checkpoints'
RESULTS_CSV = 'results.csv'
RESULTS_JSON = 'results.json'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# SentencePiece tokenizer wrapper
# -------------------------
if not os.path.exists(SP_MODEL_FILE):
    raise FileNotFoundError(f"SentencePiece model not found: {SP_MODEL_FILE}. Run preprocessing to train it.")

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_FILE)

PAD_ID = sp.pad_id() if sp.pad_id() != -1 else 0
BOS_ID = sp.bos_id() if sp.bos_id() != -1 else None
EOS_ID = sp.eos_id() if sp.eos_id() != -1 else None
UNK_ID = sp.unk_id() if sp.unk_id() != -1 else None

VOCAB_SIZE = sp.get_piece_size()

def encode_text(text, add_bos=True, add_eos=True, max_len=None):
    ids = sp.encode(text, out_type=int)
    if add_bos and BOS_ID is not None:
        ids = [BOS_ID] + ids
    if add_eos and EOS_ID is not None:
        ids = ids + [EOS_ID]
    if max_len is not None:
        ids = ids[:max_len]
    return ids

def decode_ids(ids):
    ids = [i for i in ids if i != PAD_ID]
    return sp.decode(ids)

# -------------------------
# Dataset / DataLoader
# -------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, csv_file, input_col='input_text', target_col='target_text', max_target_len=MAX_TARGET_LEN):
        self.df = pd.read_csv(csv_file)
        self.input_col = input_col
        self.target_col = target_col
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = str(row.get(self.input_col, ''))
        tgt = str(row.get(self.target_col, ''))
        src_ids = encode_text(src, add_bos=False, add_eos=False)
        tgt_ids = encode_text(tgt, add_bos=True, add_eos=True, max_len=self.max_target_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_batch(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    src_padded = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD_ID)
    tgt_padded = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD_ID)
    return src_padded.to(DEVICE), torch.tensor(src_lens, dtype=torch.long).to(DEVICE), tgt_padded.to(DEVICE), torch.tensor(tgt_lens, dtype=torch.long).to(DEVICE)

# -------------------------
# Transformer building blocks (top-level)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


def attention(q, k, v, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (B, S_q, D), (B, S_k, D), (B, S_v, D)
        nbatches = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        # linear projections
        q = self.linears[0](query)  # (B, S_q, D)
        k = self.linears[1](key)    # (B, S_k, D)
        v = self.linears[2](value)  # (B, S_v, D)

        # reshape to (B, h, S_*, d_k) for each
        q = q.view(nbatches, q_len, self.h, self.d_k).transpose(1, 2)
        k = k.view(nbatches, k_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(nbatches, v_len, self.h, self.d_k).transpose(1, 2)

        if mask is not None:
            # mask should be broadcastable to (B, 1, S_q, S_k) or similar
            # if mask is (B,1,1,S_k) or (B,1,S_q,S_k) it will broadcast appropriately
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask

        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)

        # x: (B, h, S_q, d_k) -> (B, S_q, h*d_k)
        x = x.transpose(1, 2).contiguous().view(nbatches, q_len, self.h * self.d_k)

        # final linear projection
        out = self.linears[-1](x)
        # sanity check: output should match query's shape (B, S_q, D)
        assert out.shape == query.shape, f"MHA output shape {out.shape} != query shape {query.shape}"
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        att = self.self_attn(x, x, x, mask=src_mask)
        # att expected shape (B, S, D)
        assert att.shape == x.shape, f"Self-attention output shape {att.shape} doesn't match input {x.shape}"
        x = x + self.dropout(att)
        x = self.norm1(x)
        ff = self.feed_forward(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        att1 = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(att1)
        x = self.norm1(x)
        att2 = self.src_attn(x, memory, memory, mask=src_mask)
        x = x + self.dropout(att2)
        x = self.norm2(x)
        ff = self.feed_forward(x)
        x = x + self.dropout(ff)
        x = self.norm3(x)
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=NHEADS, num_encoder_layers=ENC_LAYERS,
                 num_decoder_layers=DEC_LAYERS, dim_feedforward=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoder = PositionalEncoding(d_model)

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        src_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, self_attn, feed_forward, dropout), num_encoder_layers)
        self.decoder = Decoder(DecoderLayer(d_model, self_attn, src_attn, feed_forward, dropout), num_decoder_layers)

        self.out = nn.Linear(d_model, vocab_size)

    def make_src_mask(self, src):
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
        T = tgt.size(1)
        subsequent_mask = torch.triu(torch.ones((1, 1, T, T), device=tgt.device), diagonal=1).bool()
        return tgt_pad_mask & ~subsequent_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_mask=src_mask)
        out = self.decoder(tgt_emb, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.out(out)
        return logits

    def encode(self, src):
        src_mask = self.make_src_mask(src)
        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask=src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        return self.decoder(tgt_emb, memory, src_mask=src_mask, tgt_mask=tgt_mask)

# -------------------------
# Metrics (simple implementations)
# -------------------------
def ngram_counts(segment, n):
    return Counter([tuple(segment[i:i+n]) for i in range(len(segment)-n+1)]) if len(segment) >= n else Counter()


def save_architecture(model: nn.Module, path: str):
    """Save a small JSON describing the model architecture/hyperparameters."""
    arch = {}
    # Try to extract common attributes
    arch['d_model'] = getattr(model, 'd_model', None)
    arch['vocab_size'] = getattr(model, 'out', None) and getattr(model.out, 'out_features', None) or getattr(model, 'vocab_size', None)
    arch['nhead'] = NHEADS
    arch['encoder_layers'] = ENC_LAYERS
    arch['decoder_layers'] = DEC_LAYERS
    arch['embed_dim'] = EMBED_DIM
    arch['ffn_dim'] = FFN_DIM
    arch['dropout'] = DROPOUT
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(arch, f, indent=2)

def corpus_bleu(references, hypotheses, max_n=4):
    precisions = [0.0] * max_n
    total_hyp_len = 0
    total_ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        total_hyp_len += max(1, len(hyp))
        total_ref_len += len(ref)
        for n in range(1, max_n+1):
            ref_counts = ngram_counts(ref, n)
            hyp_counts = ngram_counts(hyp, n)
            matched = 0
            for ng, cnt in hyp_counts.items():
                matched += min(cnt, ref_counts.get(ng, 0))
            precisions[n-1] += matched
    precisions = [p / total_hyp_len if total_hyp_len > 0 else 0.0 for p in precisions]
    import math as _math
    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = _math.exp(sum(_math.log(p) for p in precisions) / max_n)
    bp = 1.0
    if total_hyp_len <= total_ref_len:
        bp = _math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0
    return bp * geo_mean


def rouge_l_score(references, hypotheses):
    def lcs(a, b):
        la, lb = len(a), len(b)
        dp = [0] * (lb+1)
        for i in range(la):
            prev = 0
            for j in range(lb):
                temp = dp[j+1]
                if a[i] == b[j]:
                    dp[j+1] = prev + 1
                else:
                    dp[j+1] = max(dp[j+1], dp[j])
                prev = temp
        return dp[lb]

    scores = []
    for ref, hyp in zip(references, hypotheses):
        l = lcs(ref, hyp)
        prec = l / len(hyp) if len(hyp) > 0 else 0.0
        rec = l / len(ref) if len(ref) > 0 else 0.0
        if prec + rec == 0:
            f = 0.0
        else:
            f = (2 * prec * rec) / (prec + rec)
        scores.append(f)
    return sum(scores) / len(scores) if scores else 0.0


def chrf_score(references, hypotheses, max_order=6):
    def char_ngrams(s, n):
        return Counter([s[i:i+n] for i in range(len(s)-n+1)]) if len(s) >= n else Counter()

    precisions = []
    recalls = []
    for ref, hyp in zip(references, hypotheses):
        ref_s = ''.join(ref)
        hyp_s = ''.join(hyp)
        p_total = 0
        r_total = 0
        for n in range(1, max_order+1):
            r_counts = char_ngrams(ref_s, n)
            h_counts = char_ngrams(hyp_s, n)
            match = sum(min(h_counts[g], r_counts.get(g, 0)) for g in h_counts)
            h_sum = sum(h_counts.values())
            r_sum = sum(r_counts.values())
            p_total += (match / h_sum) if h_sum > 0 else 0
            r_total += (match / r_sum) if r_sum > 0 else 0
        precisions.append(p_total / max_order)
        recalls.append(r_total / max_order)
    p = sum(precisions) / len(precisions) if precisions else 0
    r = sum(recalls) / len(recalls) if recalls else 0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# -------------------------
# Training and evaluation
# -------------------------
def train(batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, decode_method='greedy', beam_size=4,
          early_stopping=True, patience=3, min_delta=1e-4):
    """Train the seq2seq model. Parameters can override module-level defaults."""
    train_ds = Seq2SeqDataset(TRAIN_CSV)
    val_ds = Seq2SeqDataset(VAL_CSV)
    test_ds = Seq2SeqDataset(TEST_CSV)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = TransformerSeq2Seq(VOCAB_SIZE, d_model=EMBED_DIM, nhead=NHEADS,
                               num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS,
                               dim_feedforward=FFN_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=BETAS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_bleu = 0.0
    history = []
    epochs_since_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for src, src_lens, tgt, tgt_lens in tqdm(train_loader, desc=f"Epoch {epoch} - train", leave=False):
            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]
            logits = model(src, decoder_input)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), labels.contiguous().view(B*T))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader)
        bleu = val_metrics['bleu']
        perp = val_metrics['perplexity']

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'bleu': bleu,
            'rouge_l': val_metrics['rouge_l'],
            'chrf': val_metrics['chrf'],
            'perplexity': perp,
        })

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, BLEU={bleu:.4f}, PPL={perp:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_epoch{epoch}.pth')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
        # save architecture alongside the checkpoint
        arch_path = os.path.join(CHECKPOINT_DIR, f'model_epoch{epoch}_arch.json')
        save_architecture(model, arch_path)

        # check for improvement with min_delta
        if bleu > best_bleu + min_delta:
            best_bleu = bleu
            epochs_since_improve = 0
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, best_path)
            save_architecture(model, os.path.join(CHECKPOINT_DIR, 'best_model_arch.json'))
        else:
            epochs_since_improve += 1

        # early stopping
        if early_stopping and epochs_since_improve >= patience:
            print(f"Early stopping: no improvement in BLEU for {epochs_since_improve} epochs (patience={patience}).")
            break

    pd.DataFrame(history).to_csv(RESULTS_CSV, index=False)
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Build a concise summary JSON matching the requested structure
    if history:
        all_bleu_scores = [float(h.get('bleu', 0.0)) for h in history]
        all_rouge_scores = [float(h.get('rouge_l', 0.0)) for h in history]
        all_chrf_scores = [float(h.get('chrf', 0.0)) for h in history]
        all_perplexities = [float(h.get('perplexity', float('inf'))) for h in history]

        # final epoch metrics
        final_train_loss = float(history[-1].get('train_loss', 0.0))
        final_val_loss = float(history[-1].get('val_loss', 0.0))

        # best by BLEU
        best_idx = int(max(range(len(all_bleu_scores)), key=lambda i: all_bleu_scores[i]))
        best_bleu = float(all_bleu_scores[best_idx])
        best_rouge = float(all_rouge_scores[best_idx]) if best_idx < len(all_rouge_scores) else 0.0
        best_chrf = float(all_chrf_scores[best_idx]) if best_idx < len(all_chrf_scores) else 0.0
        best_perplexity = float(all_perplexities[best_idx]) if best_idx < len(all_perplexities) else float('inf')

        summary = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_bleu': best_bleu,
            'best_rouge': best_rouge,
            'best_chrf': best_chrf,
            'best_perplexity': best_perplexity,
            'total_epochs': len(history),
            'all_bleu_scores': all_bleu_scores,
            'all_rouge_scores': all_rouge_scores,
            'all_chrf_scores': all_chrf_scores,
            'all_perplexities': all_perplexities,
        }

        results_summary_path = os.path.join('.', 'results_summary.json')
        with open(results_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    # After training, also save a few qualitative examples for human evaluation
    try:
        save_human_eval_samples(model, test_ds, n=20, out_file='human_eval.csv', decode_method=decode_method, beam_size=beam_size)
        print('Saved human evaluation samples to human_eval.csv')
    except Exception:
        # not critical
        pass


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0
    refs = []
    hyps = []
    with torch.no_grad():
        for src, src_lens, tgt, tgt_lens in tqdm(loader, desc="Evaluating", leave=False):
            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]
            logits = model(src, decoder_input)
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), labels.contiguous().view(B*T))
            total_loss += loss.item() * src.size(0)

            memory = model.encode(src)
            for i in range(src.size(0)):
                pred_ids = [BOS_ID] if BOS_ID is not None else []
                for _ in range(MAX_TARGET_LEN):
                    cur = torch.tensor(pred_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                    cur_padded = nn.utils.rnn.pad_sequence([cur.squeeze(0)], batch_first=True, padding_value=PAD_ID).to(DEVICE)
                    logits_step = model(src[i:i+1], cur_padded)
                    next_token_logits = logits_step[0, -1, :]
                    next_id = int(torch.argmax(next_token_logits).item())
                    pred_ids.append(next_id)
                    if EOS_ID is not None and next_id == EOS_ID:
                        break
                hyps.append([str(tok) for tok in pred_ids if tok not in (PAD_ID, BOS_ID)])
                ref_seq = tgt[i].tolist()
                refs.append([str(tok) for tok in ref_seq if tok not in (PAD_ID, BOS_ID, EOS_ID)])

    avg_loss = total_loss / len(loader.dataset)
    bleu = corpus_bleu(refs, hyps)
    rouge_l = rouge_l_score(refs, hyps)
    chrf = chrf_score(refs, hyps)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return {'loss': avg_loss, 'bleu': bleu, 'rouge_l': rouge_l, 'chrf': chrf, 'perplexity': perplexity}


def generate_greedy(model, src_tensor, max_len=MAX_TARGET_LEN):
    """Generate a prediction (list of token ids) for a single src tensor using greedy decoding."""
    model.eval()
    with torch.no_grad():
        src = src_tensor.unsqueeze(0).to(DEVICE) if src_tensor.dim() == 1 else src_tensor.to(DEVICE)
        memory = model.encode(src)
        pred_ids = [BOS_ID] if BOS_ID is not None else []
        for _ in range(max_len):
            cur = torch.tensor(pred_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            cur_padded = nn.utils.rnn.pad_sequence([cur.squeeze(0)], batch_first=True, padding_value=PAD_ID).to(DEVICE)
            logits_step = model(src, cur_padded)
            next_token_logits = logits_step[0, -1, :]
            next_id = int(torch.argmax(next_token_logits).item())
            pred_ids.append(next_id)
            if EOS_ID is not None and next_id == EOS_ID:
                break
    return pred_ids


def generate_beam(model, src_tensor, beam_size=4, max_len=MAX_TARGET_LEN, length_penalty=1.0):
    """Beam search generation for a single source tensor.

    Returns best sequence of token ids (including BOS/EOS if present).
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        src = src_tensor.unsqueeze(0).to(device) if src_tensor.dim() == 1 else src_tensor.to(device)
        memory = model.encode(src)

        # Each beam: (tokens_list, score, finished)
        init_tokens = [BOS_ID] if BOS_ID is not None else []
        beams = [(init_tokens, 0.0, False)]

        for _step in range(max_len):
            all_candidates = []
            # Prepare batch of current sequences
            seqs = [torch.tensor(b[0], dtype=torch.long, device=device).unsqueeze(0) for b in beams]
            # If any beam has empty seq list, replace with tensor([PAD_ID]) to avoid errors
            seqs = [s if s.numel() > 0 else torch.tensor([PAD_ID], dtype=torch.long, device=device).unsqueeze(0) for s in seqs]
            cur_padded = nn.utils.rnn.pad_sequence([s.squeeze(0) for s in seqs], batch_first=True, padding_value=PAD_ID)
            # Repeat src for each beam
            src_rep = src.repeat(len(beams), 1)
            logits = model(src_rep, cur_padded)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (B_beam, V)

            for i, (tokens, score, finished) in enumerate(beams):
                if finished:
                    # carry forward finished beams
                    all_candidates.append((tokens, score, True))
                    continue
                probs = log_probs[i]  # tensor of size V
                topk = torch.topk(probs, k=min(beam_size, probs.size(0)))
                topk_vals = topk.values.cpu().tolist()
                topk_idx = topk.indices.cpu().tolist()
                for val, idx in zip(topk_vals, topk_idx):
                    new_tokens = tokens + [int(idx)]
                    new_score = score + float(val)
                    new_finished = (EOS_ID is not None and idx == EOS_ID)
                    all_candidates.append((new_tokens, new_score, new_finished))

            # Select top beam_size candidates
            ordered = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** length_penalty if length_penalty != 1.0 else 1.0), reverse=True)
            beams = ordered[:beam_size]

            # If all beams finished, stop
            if all(b[2] for b in beams):
                break

        # Choose best finished beam (prefer finished), otherwise top-scoring
        finished_beams = [b for b in beams if b[2]]
        best = finished_beams[0] if finished_beams else beams[0]
        return best[0]


def generate(model, src_tensor, method='greedy', beam_size=4, max_len=MAX_TARGET_LEN):
    if method == 'greedy':
        return generate_greedy(model, src_tensor, max_len=max_len)
    elif method == 'beam':
        return generate_beam(model, src_tensor, beam_size=beam_size, max_len=max_len)
    else:
        raise ValueError(f'Unknown generation method: {method}')


def save_human_eval_samples(model, dataset: Seq2SeqDataset, n=20, out_file='human_eval.csv', decode_method='greedy', beam_size=4):
    """Save n examples from the dataset to a CSV with fields for human ratings.

    CSV columns: input_text, reference, prediction, fluency, relevance, adequacy, notes
    Fluency/Relevance/Adequacy are left blank for human annotators to fill (1-5 scale).
    """
    n = min(n, len(dataset))
    rows = []
    for idx in range(n):
        row = dataset.df.iloc[idx]
        input_text = str(row.get(dataset.input_col, ''))
        ref = str(row.get(dataset.target_col, ''))
        src_ids = encode_text(input_text, add_bos=False, add_eos=False)
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        pred_ids = generate(model, src_tensor, method=decode_method, beam_size=beam_size, max_len=MAX_TARGET_LEN)
        pred_text = decode_ids(pred_ids)
        rows.append({
            'input_text': input_text,
            'reference': ref,
            'prediction': pred_text,
            'fluency': '',
            'relevance': '',
            'adequacy': '',
            'notes': ''
        })

    fieldnames = ['input_text', 'reference', 'prediction', 'fluency', 'relevance', 'adequacy', 'notes']
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer seq2seq')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--only-generate', action='store_true', help='Skip training and generate human eval samples from best checkpoint')
    parser.add_argument('--generate-n', type=int, default=20, help='Number of examples to generate for human eval')
    parser.add_argument('--decoding', choices=['greedy', 'beam'], default='greedy', help='Decoding method for generation')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam size when using beam decoding')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation BLEU')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (epochs)')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Minimum BLEU improvement to reset early stopping')
    args = parser.parse_args()

    start = time.time()
    if args.only_generate:
        # load best model
        best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError('Best model not found, cannot generate samples')
        # construct model and load state
        model = TransformerSeq2Seq(VOCAB_SIZE, d_model=EMBED_DIM, nhead=NHEADS,
                                   num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS,
                                   dim_feedforward=FFN_DIM, dropout=DROPOUT).to(DEVICE)
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        test_ds = Seq2SeqDataset(TEST_CSV)
        save_human_eval_samples(model, test_ds, n=args.generate_n, out_file='human_eval.csv', decode_method=args.decoding, beam_size=args.beam_size)
        print('Saved human evaluation samples to human_eval.csv')
    else:
        train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, decode_method=args.decoding, beam_size=args.beam_size, early_stopping=args.early_stopping, patience=args.patience, min_delta=args.min_delta)

    print('Done.')
