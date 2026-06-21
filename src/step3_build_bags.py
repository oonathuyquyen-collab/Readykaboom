"""STEP 3 - Build TMIL bags for the PTXPhish population (same schema as train_bags.pkl).
Reuses the existing counterparty vocab so IDs align with the model trained in-domain.
Each bag: account, input_ids (counterparty vocab id), input_io (1=OUT,2=IN),
input_amounts (ETH), delta_ts (s), label, length, gt_idx (positions of GT phishing txs),
ntx_full (full history size, for activity-stratified eval), direction.
Applies the BERT4ETH account filter 3<=ntx<=10000.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab  # for unpickling existing vocab

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SRC_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "drive_folder", "tmil_eth", "data", "bert4eth", "vocab.pkl")
MAXLEN = 100
MIN_TX, MAX_TX = 3, 10000

def load_vocab():
    with open(SRC_VOCAB, "rb") as f:
        return pickle.load(f)

def seq_from_txs(txs, account):
    seq = []
    for t in txs:
        try:
            frm = t["from"].lower(); to = (t.get("to") or "").lower()
            ts = int(t["timeStamp"]); val = int(t["value"]) / 1e18
            h = t["hash"].lower()
        except Exception:
            continue
        if frm == account:
            seq.append((ts, val, 1, to, h))       # OUT
        elif to == account:
            seq.append((ts, val, 2, frm, h))      # IN
    seq.sort(key=lambda x: x[0])
    return seq

def build_bag(account, txs, label, vocab, gt_hashes=None):
    seq = seq_from_txs(txs, account)
    ntx_full = len(seq)
    if ntx_full < MIN_TX or ntx_full > MAX_TX:
        return None
    gt_hashes = set(h.lower() for h in (gt_hashes or []))
    # choose window: positives -> 100-window ending at latest GT tx (guarantees GT present)
    if gt_hashes:
        gt_positions = [i for i, s in enumerate(seq) if s[4] in gt_hashes]
        if not gt_positions:
            return None  # GT tx not in this account's external history
        end = max(gt_positions) + 1
        start = max(0, end - MAXLEN)
        window = seq[start:end]
    else:
        window = seq[-MAXLEN:]
        start = len(seq) - len(window)
    input_ids, input_io, input_amounts, delta_ts, hashes = [], [], [], [], []
    last = None
    for (ts, val, io, cp, h) in window:
        input_ids.append(vocab.get_id(cp))
        input_io.append(io); input_amounts.append(val)
        delta_ts.append(0 if last is None else ts - last); last = ts
        hashes.append(h)
    gt_idx = [i for i, h in enumerate(hashes) if h in gt_hashes]
    return {
        "account": account, "input_ids": input_ids, "input_io": input_io,
        "input_amounts": input_amounts, "delta_ts": delta_ts, "label": label,
        "length": len(window), "gt_idx": gt_idx, "ntx_full": ntx_full,
        "hashes": hashes,
    }

def main():
    vocab = load_vocab()
    print("vocab size:", len(vocab.token2id))
    pos_hist = json.load(open(os.path.join(DATA, "step2_pos_hist.json")))
    neg_hist = json.load(open(os.path.join(DATA, "step2_neg_hist.json")))
    gt_map = json.load(open(os.path.join(DATA, "step2_gt_map.json")))

    pos_bags, neg_bags = [], []
    drop_nogt = 0
    for acc, txs in pos_hist.items():
        b = build_bag(acc, txs, 1, vocab, gt_hashes=gt_map.get(acc, []))
        if b is None:
            drop_nogt += 1; continue
        pos_bags.append(b)
    for acc, txs in neg_hist.items():
        b = build_bag(acc, txs, 0, vocab, gt_hashes=None)
        if b: neg_bags.append(b)

    print(f"POS bags kept: {len(pos_bags)} (dropped {drop_nogt}); NEG bags: {len(neg_bags)}")
    allb = pos_bags + neg_bags
    with open(os.path.join(DATA, "ptx_bags.pkl"), "wb") as f:
        pickle.dump(allb, f)
    # quick stats
    import numpy as np
    print("pos length stats:", np.percentile([b['length'] for b in pos_bags],[0,50,100]) if pos_bags else None)
    print("pos with >=1 gt:", sum(1 for b in pos_bags if b['gt_idx']))
    print("saved ptx_bags.pkl:", len(allb))

if __name__ == "__main__":
    main()
