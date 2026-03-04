from __future__ import annotations

import re
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#setup for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

#Token and Vocab
SPECIALS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

def tokenize(s: str) -> List[str]:
    s = s.lower().strip()
    # keep letters + common french accents + apostrophe + whitespace; replace everything else with space
    s = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.split()

class Vocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build(self, sentences: List[str]):
        freq: Dict[str, int] = {}
        for s in sentences:
            for t in tokenize(s):
                freq[t] = freq.get(t, 0) + 1

        self.itos = SPECIALS[:]
        for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= self.min_freq and tok not in self.itos:
                self.itos.append(tok)

        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, s: str) -> List[int]:
        ids = [self.stoi["<SOS>"]]
        for t in tokenize(s):
            ids.append(self.stoi.get(t, self.stoi["<UNK>"]))
        ids.append(self.stoi["<EOS>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else "<UNK>"
            if tok in ("<SOS>", "<PAD>"):
                continue
            if tok == "<EOS>":
                break
            out.append(tok)
        return " ".join(out)

    @property
    def pad_id(self) -> int: return self.stoi["<PAD>"]

    @property
    def sos_id(self) -> int: return self.stoi["<SOS>"]

    @property
    def eos_id(self) -> int: return self.stoi["<EOS>"]

    def __len__(self) -> int:
        return len(self.itos)

#Data
def load_tsv_pairs(path: str, max_pairs: int | None = None) -> List[Tuple[str, str]]:
    """
    Reads tab-separated lines:
      EN \t FR \t (maybe extra columns...)
    Returns pairs as (FR, EN) for FR->EN translation.
    """
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            en = parts[0].strip()
            fr = parts[1].strip()
            if not en or not fr:
                continue

            pairs.append((fr, en))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], fr_vocab: Vocab, en_vocab: Vocab):
        self.pairs = pairs
        self.fr_vocab = fr_vocab
        self.en_vocab = en_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        fr, en = self.pairs[idx]
        src = torch.tensor(self.fr_vocab.encode(fr), dtype=torch.long)
        tgt = torch.tensor(self.en_vocab.encode(en), dtype=torch.long)
        return src, tgt

def collate_batch(batch, pad_src: int, pad_tgt: int):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(x) for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgts], dtype=torch.long)

    max_s = int(src_lens.max().item())
    max_t = int(tgt_lens.max().item())

    src_pad = torch.full((len(batch), max_s), pad_src, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_t), pad_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t

    return src_pad, src_lens, tgt_pad, tgt_lens

##Model
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hid: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        emb = self.embedding(src_ids)  # [B,S,E]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return enc_out, (h, c)

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention:
      e_{t,i} = v^T tanh(W h_i + U s_t)
      alpha_{t,i} = softmax(e_{t,i})
      c_t = sum_i alpha_{t,i} h_i
    """
    def __init__(self, hid: int = 512):
        super().__init__()
        self.W = nn.Linear(hid, hid, bias=False)
        self.U = nn.Linear(hid, hid, bias=False)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        proj_enc = self.W(enc_out)                    # [B,S,H]
        proj_dec = self.U(dec_h).unsqueeze(1)         # [B,1,H]
        energy = torch.tanh(proj_enc + proj_dec)      # [B,S,H]
        scores = self.v(energy).squeeze(-1)           # [B,S]
        scores = scores.masked_fill(src_mask == 0, -1e9)
        attn_w = torch.softmax(scores, dim=1)         # [B,S]
        context = torch.bmm(attn_w.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return context, attn_w

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hid: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb + hid, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)

    def forward(self, prev_y: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor], context: torch.Tensor):
        emb = self.embedding(prev_y).unsqueeze(1)     # [B,1,E]
        ctx = context.unsqueeze(1)                    # [B,1,H]
        x = torch.cat([emb, ctx], dim=-1)             # [B,1,E+H]
        out, state = self.lstm(x, state)              # [B,1,H]
        logits = self.fc(out.squeeze(1))              # [B,V]
        return logits, state

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, attention: BahdanauAttention, decoder: Decoder, src_pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.src_pad_id = src_pad_id

    def src_mask(self, src_ids: torch.Tensor):
        return (src_ids != self.src_pad_id).long()

    def forward(self, src_ids, src_lens, tgt_ids, teacher_forcing: float = 0.5):
        B, T = tgt_ids.shape
        V = self.decoder.fc.out_features

        enc_out, state = self.encoder(src_ids, src_lens)
        mask = self.src_mask(src_ids)

        outputs = torch.zeros(B, T, V, device=src_ids.device)
        y_prev = tgt_ids[:, 0]  # <SOS>

        for t in range(1, T):
            dec_h = state[0].squeeze(0)               # [B,H]
            context, _ = self.attention(dec_h, enc_out, mask)
            logits, state = self.decoder(y_prev, state, context)
            outputs[:, t, :] = logits

            if random.random() < teacher_forcing:
                y_prev = tgt_ids[:, t]
            else:
                y_prev = logits.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def translate(self, fr_sentence: str, fr_vocab: Vocab, en_vocab: Vocab, max_steps: int = 40) -> str:
        self.eval()
        src_ids = torch.tensor([fr_vocab.encode(fr_sentence)], dtype=torch.long, device=DEVICE)
        src_lens = torch.tensor([src_ids.size(1)], dtype=torch.long, device=DEVICE)

        enc_out, state = self.encoder(src_ids, src_lens)
        mask = self.src_mask(src_ids)

        y_prev = torch.tensor([en_vocab.sos_id], dtype=torch.long, device=DEVICE)
        out_ids: List[int] = []

        for _ in range(max_steps):
            dec_h = state[0].squeeze(0)               # [1,H]
            context, _ = self.attention(dec_h, enc_out, mask)
            logits, state = self.decoder(y_prev, state, context)
            y_prev = logits.argmax(dim=1)

            tok = int(y_prev.item())
            if tok == en_vocab.eos_id:
                break
            out_ids.append(tok)

        return en_vocab.decode(out_ids)

#Train
def main():
    TRAIN_PATH = "data/eng-fra_train.tsv"
    TEST_PATH  = "data/eng-fra_test.tsv"

    #Anyone trying to run the model remember to keep the hid, batch and epochs low if you are not on PC
    EMB = 256
    HID = 256
    BATCH = 64
    EPOCHS = 3
    LR = 1e-3
    TEACHER_FORCING = 0.5

    train_pairs = load_tsv_pairs(TRAIN_PATH)
    test_pairs = load_tsv_pairs(TEST_PATH)

    if not train_pairs or not test_pairs:
        raise RuntimeError("Could not load train/test TSV files. Check paths and tab-separated format.")

    fr_sentences = [fr for fr, _ in train_pairs]
    en_sentences = [en for _, en in train_pairs]

    fr_vocab = Vocab(min_freq=2)
    en_vocab = Vocab(min_freq=2)
    fr_vocab.build(fr_sentences)
    en_vocab.build(en_sentences)

    train_ds = PairDataset(train_pairs, fr_vocab, en_vocab)
    test_ds = PairDataset(test_pairs, fr_vocab, en_vocab)

    print(f"Device: {DEVICE}")
    print(f"Training config -> epochs={EPOCHS}, batch={BATCH}, hid={HID}, emb={EMB}, lr={LR}")
    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")   
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        collate_fn=lambda b: collate_batch(b, fr_vocab.pad_id, en_vocab.pad_id)
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        collate_fn=lambda b: collate_batch(b, fr_vocab.pad_id, en_vocab.pad_id)
    )

    model = Seq2Seq(
        Encoder(len(fr_vocab), EMB, HID),
        BahdanauAttention(HID),
        Decoder(len(en_vocab), EMB, HID),
        src_pad_id=fr_vocab.pad_id
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=en_vocab.pad_id)

    def eval_loss() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch_idx, (src, src_lens, tgt, _) in enumerate(train_loader):
                src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)

            opt.zero_grad()
            out = model(src, src_lens, tgt, teacher_forcing=TEACHER_FORCING)

            logits = out[:, 1:, :].reshape(-1, out.size(-1))
            gold = tgt[:, 1:].reshape(-1)

            loss = crit(logits, gold)

            total += loss.item()
            n += 1

         # Print every 100 batches so you we see movement
            if batch_idx % 100 == 0:
             print(f"Epoch {ep} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}")
        return total / max(1, n)

    for ep in range(1, EPOCHS + 1):
        model.train()
        total, n = 0.0, 0

        print(f"\n--- Epoch {ep}/{EPOCHS} ---")

        for batch_idx, (src, src_lens, tgt, _) in enumerate(train_loader):
            src, src_lens, tgt = src.to(DEVICE), src_lens.to(DEVICE), tgt.to(DEVICE)

            opt.zero_grad()
            out = model(src, src_lens, tgt, teacher_forcing=TEACHER_FORCING)

            logits = out[:, 1:, :].reshape(-1, out.size(-1))
            gold = tgt[:, 1:].reshape(-1)

            loss = crit(logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += loss.item()
            n += 1

            # print every 50 batches so you can SEE it training
            if batch_idx % 50 == 0:
                avg = total / max(1, n)
                print(f"Epoch {ep} | Batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f} | avg={avg:.4f}")

        print(f"Epoch {ep:02d} DONE | train_loss={total/max(1,n):.4f} | test_loss={eval_loss():.4f}")

        sample_fr, _ = random.choice(test_pairs)
        print("FR:", sample_fr)
        print("EN:", model.translate(sample_fr, fr_vocab, en_vocab))
        print("-" * 60)

if __name__ == "__main__":
    main()