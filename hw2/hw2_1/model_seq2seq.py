"""
Adarsha Neupane
HW_2 of CPSC8430 Deep Learning
Date: 10.03.2025

- Seq2Seq video captioning with attention
- Scheduled sampling (linear decay)
- Beam search decoding
- train / infer subcommands

Expected data layout under <DATA_DIR>:
  training_label.json
  testing_label.json
  training_data/feat/*.avi.npy
  testing_data/feat/*.avi.npy
  testing_data/id.txt

"""

#Import necessary libraries
import os
import re
import json
import math
import random
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random, torch, numpy as np

# Giving seed to ensure reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Vocab
SPECIALS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

class Vocab:
    def __init__(self, min_count=3): # min_count should be greater than 3
        self.min_count = min_count
        self.word2idx = dict(SPECIALS)
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.freqs = Counter()

    def add_sentence(self, tokens):
        self.freqs.update(tokens)

    def build(self):
        for w, c in self.freqs.items():
            if c > self.min_count and w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def encode(self, tokens, add_bos=True, add_eos=True):
        ids = []
        if add_bos:
            ids.append(self.bos_idx)
        for t in tokens:
            # returns the index of word if it is in the dict, otherwise the index of <UNK>
            ids.append(self.word2idx.get(t, self.unk_idx)) 
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids):
        return [self.idx2word.get(i, "<UNK>") for i in ids]

    @property
    def pad_idx(self): return self.word2idx["<PAD>"]
    @property
    def bos_idx(self): return self.word2idx["<BOS>"]
    @property
    def eos_idx(self): return self.word2idx["<EOS>"]
    @property
    def unk_idx(self): return self.word2idx["<UNK>"]

    def __len__(self): return len(self.word2idx)


# Tokenizer
_token_pat = re.compile(r"[^a-z0-9' ]+")

def simple_tokenize(s: str):
    s = s.lower().strip()
    s = _token_pat.sub(" ", s)
    s = re.sub(r"\s+", " ", s) # replaces multiple spaces (\s+) with a single space
    return s.split()


# Dataset of captions from the video
class VideoCaptionDataset(Dataset):
    def __init__(self, data_dir, label_json, vocab: Vocab, max_len=30, train=True):
        self.data_dir = data_dir
        self.items = json.load(open(os.path.join(data_dir, label_json)))
        self.vocab = vocab
        self.max_len = max_len
        self.train = train

        if train:
            for it in self.items:
                for cap in it.get("caption", []):
                    toks = simple_tokenize(cap)
                    vocab.add_sentence(toks)
            vocab.build()

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        vid_id = it["id"]  
        npy_name = vid_id.replace(".avi", ".avi.npy")
        split = "training_data" if self.train else "testing_data"
        feat_path = os.path.join(self.data_dir, split, "feat", npy_name)

        feat = np.load(feat_path)  # (T, D)
        feat = torch.tensor(feat, dtype=torch.float32)

        if self.train:
            caps = it.get("caption", [""])
            cap = random.choice(caps) # randomly choose one caption from each video id
            toks = simple_tokenize(cap)
            ids = self.vocab.encode(toks)
            # truncate + ensure EOS
            ids = ids[:self.max_len]
            if ids[-1] != self.vocab.eos_idx:
                ids.append(self.vocab.eos_idx)
            cap_t = torch.tensor(ids, dtype=torch.long)
        else:
            cap_t = torch.tensor([self.vocab.bos_idx, self.vocab.eos_idx], dtype=torch.long)

        return feat, cap_t, vid_id


def pad_collate(batch, pad_idx):
    vids, caps, ids = zip(*batch)
    # videos
    T = max(v.shape[0] for v in vids) # maximum No. of time-steps/frames among all videos
    D = vids[0].shape[1] # Feature Dimension
    B = len(vids) # batch size
    vids_pad = torch.zeros(B, T, D, dtype=vids[0].dtype)
    vid_lens = []
    for i, v in enumerate(vids):
        t = v.shape[0]
        vids_pad[i, :t] = v
        vid_lens.append(t)
    vid_lens = torch.tensor(vid_lens)

    # captions
    L = max(c.shape[0] for c in caps)
    caps_pad = torch.full((B, L), pad_idx, dtype=torch.long)
    cap_lens = []
    for i, c in enumerate(caps):
        caps_pad[i, :c.shape[0]] = c
        cap_lens.append(c.shape[0])
    cap_lens = torch.tensor(cap_lens)

    return vids_pad, vid_lens, caps_pad, cap_lens, list(ids)


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=4096, hidden=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, hidden, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.hidden = hidden

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, (h, c)  


# applying Bahdanau attention
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W = nn.Linear(enc_dim, attn_dim)
        self.U = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, enc_out, dec_h, mask):
        # enc_out: (B,T,Enc)  dec_h: (B,Dec)  mask: (B,T) Bool
        scores = self.v(torch.tanh(self.W(enc_out) + self.U(dec_h).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=-1)         # (B,T)
        ctx = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)  # (B,Enc)
        return ctx, alpha

    
# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, enc_dim=256, hidden=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attn = Attention(enc_dim, hidden)
        self.rnn = nn.LSTM(
            emb_dim + enc_dim, hidden, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward_step(self, y_prev, state, enc_out, enc_mask):
        # y_prev: (B,), state: (h,c) with h shape (L,B,H)
        emb = self.embedding(y_prev).unsqueeze(1)     # (B,1,E)
        h_t = state[0][-1]                             # (B,H)
        ctx, _ = self.attn(enc_out, h_t, enc_mask)     # (B,Enc)
        rnn_in = torch.cat([emb, ctx.unsqueeze(1)], dim=-1)  # (B,1,E+Enc)
        out, state = self.rnn(rnn_in, state)           # out: (B,1,H)
        logits = self.fc(out.squeeze(1))               # (B,V)
        return logits, state


#Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, input_dim=4096, enc_hidden=256, dec_hidden=256, emb_dim=256):
        super().__init__()
        enc_dim = enc_hidden
        self.encoder = Encoder(input_dim=input_dim, hidden=enc_hidden)
        self.decoder = Decoder(vocab_size=vocab_size, emb_dim=emb_dim, enc_dim=enc_dim, hidden=dec_hidden)
        self.enc_dim = enc_dim
        self.dec_hidden = dec_hidden

    def make_mask(self, lengths, max_len):
        idxs = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return (idxs < lengths.unsqueeze(1))  # (B,T) bool

    def forward(self, vids, vid_lens, caps, teacher_ratio=1.0, bos_idx=1, eos_idx=2, pad_idx=0):
        # Encode
        enc_out, (h, c) = self.encoder(vids, vid_lens)
        B, T, E = enc_out.shape
        enc_mask = self.make_mask(vid_lens, T)
        state = (h, c)

        # Decode
        max_len = caps.size(1)
        logits = []
        y_t = caps[:, 0]  # <BOS>
        for t in range(1, max_len):
            step_logits, state = self.decoder.forward_step(y_t, state, enc_out, enc_mask)
            logits.append(step_logits.unsqueeze(1))
            # scheduled sampling
            if random.random() < teacher_ratio:
                y_t = caps[:, t]
            else:
                y_t = step_logits.argmax(-1)
        return torch.cat(logits, dim=1)  # (B, L-1, V)


# Schedule sampling
def linear_decay(step, start=1.0, end=0.6, total=20):
    # teacher-forcing ratio decays from start -> end
    frac = min(1.0, step / max(1, total))
    return max(end, start - (start - end) * frac)


# Beam search
class Hyp:
    def __init__(self, tokens, logp, state):
        self.tokens = tokens
        self.logp = logp
        self.state = state
    def __lt__(self, other):
        return self.logp < other.logp

@torch.no_grad()
def beam_decode(model, vid, vid_len, bos_idx, eos_idx, beam=5, max_len=30):
    device = vid.device
    enc_out, (h, c) = model.encoder(vid.unsqueeze(0), vid_len.unsqueeze(0))
    T = enc_out.size(1)
    enc_mask = model.make_mask(vid_len.unsqueeze(0), T)

    beams = [Hyp(tokens=[bos_idx], logp=0.0, state=(h, c))]
    finished = []

    for _ in range(max_len):
        new_beams = []
        for hyp in beams:
            y_prev = torch.tensor([hyp.tokens[-1]], device=device)
            logits, state = model.decoder.forward_step(y_prev, hyp.state, enc_out, enc_mask)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topv, topi = torch.topk(log_probs, beam)
            for lp, tok in zip(topv.tolist(), topi.tolist()):
                new_tokens = hyp.tokens + [tok]
                new_logp = hyp.logp + lp
                if tok == eos_idx:
                    finished.append(Hyp(new_tokens, new_logp, state))
                else:
                    new_beams.append(Hyp(new_tokens, new_logp, state))
        # prune
        new_beams.sort(key=lambda h: h.logp, reverse=True)
        beams = new_beams[:beam]
        if len(finished) >= beam:
            break

    if not finished:
        finished = beams
    best = max(finished, key=lambda h: h.logp)
    return best.tokens


# Training & Inference entry points
def train_main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    vocab = Vocab(min_count=3)
    train_set = VideoCaptionDataset(args.data_dir, "training_label.json", vocab, max_len=args.max_len, train=True)
    pad_idx = vocab.pad_idx

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, pad_idx)
    )

    model = Seq2Seq(
        vocab_size=len(vocab),
        input_dim=args.input_dim,
        enc_hidden=args.hidden,
        dec_hidden=args.hidden,
        emb_dim=args.emb
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    global_step = 0
    total_steps = max(1, args.epochs * len(loader))

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for vids, vid_lens, caps, cap_lens, _ in loader:
            vids = vids.to(device)
            vid_lens = vid_lens.to(device)
            caps = caps.to(device)

            teacher_ratio = linear_decay(global_step, start=args.tf_start, end=args.tf_end, total=total_steps)
            logits = model(vids, vid_lens, caps,
                           teacher_ratio=teacher_ratio,
                           bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx, pad_idx=pad_idx)
            # targets: shift by 1 (drop BOS)
            targets = caps[:, 1: 1 + logits.size(1)]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += loss.item()
            global_step += 1

        avg = running / max(1, len(loader))
        print(f"Epoch {ep} | loss={avg:.4f} | teacher_ratio={teacher_ratio:.3f}")

        ckpt_path = os.path.join(args.save_dir, f"ep{ep}.pt")
        torch.save({"model": model.state_dict(), "vocab": vocab.word2idx}, ckpt_path)

    print("Training complete.")


@torch.no_grad()
def infer_main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)

    word2idx = ckpt["vocab"]
    vocab = Vocab(min_count=3)
    vocab.word2idx = word2idx
    vocab.idx2word = {i: w for w, i in word2idx.items()}

    model = Seq2Seq(vocab_size=len(vocab), input_dim=args.input_dim,
                    enc_hidden=args.hidden, dec_hidden=args.hidden, emb_dim=args.emb).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = [l.strip() for l in open(args.ids_txt)]
    out_f = open(args.out_txt, "w")

    for vid_id in ids:
        npy = os.path.join(args.data_dir, "testing_data", "feat", vid_id.replace(".avi", ".avi.npy"))
        feat = torch.tensor(np.load(npy), dtype=torch.float32, device=device)
        tokens = beam_decode(
            model, feat, torch.tensor(feat.size(0), device=device),
            bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx, beam=args.beam, max_len=args.max_len
        )
        # drop BOS/EOS/PAD
        words = [vocab.idx2word.get(t, "<UNK>") for t in tokens
                 if t not in (vocab.bos_idx, vocab.eos_idx, vocab.pad_idx)]
        caption = " ".join(words)
        out_f.write(f"{vid_id},{caption}\n")

    out_f.close()
    print(f"Wrote predictions to {args.out_txt}")


# Command Line Interface
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    pt = sub.add_parser("train")
    pt.add_argument("--data_dir", required=True)
    pt.add_argument("--save_dir", default="seq2seq_model")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--input_dim", type=int, default=4096)
    pt.add_argument("--hidden", type=int, default=256)
    pt.add_argument("--emb", type=int, default=256)
    pt.add_argument("--max_len", type=int, default=30)
    pt.add_argument("--tf_start", type=float, default=1.0)  # teacher forcing ratio start
    pt.add_argument("--tf_end", type=float, default=0.6)    # teacher forcing ratio end

    # Infer
    pi = sub.add_parser("infer")
    pi.add_argument("--data_dir", required=True)
    pi.add_argument("--ids_txt", required=True)
    pi.add_argument("--ckpt_path", required=True)
    pi.add_argument("--out_txt", required=True)
    pi.add_argument("--beam", type=int, default=5)
    pi.add_argument("--max_len", type=int, default=30)
    pi.add_argument("--device", default="cuda")
    pi.add_argument("--input_dim", type=int, default=4096)
    pi.add_argument("--hidden", type=int, default=256)
    pi.add_argument("--emb", type=int, default=256)

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "infer":
        infer_main(args)