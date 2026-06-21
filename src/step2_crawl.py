"""STEP 2 - Crawl full external tx history for the PTXPhish account population.
Positives: unique scammer accounts from CLEAN audit rows (v1 scope = categories whose
PTXPhish tx is a direct on-chain instance in the account's own history).
Negatives: senders (`from`) of benign tx hashes (KOL + DeFi developer).
Each account -> account/txlist (up to `OFFSET` external txs), cached in sqlite.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import etherscan as es

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_BENIGN = int(os.environ.get("MAX_BENIGN", "1600"))   # match positive scale
MAX_PAGES = 1
OFFSET = 10000

def crawl_accounts(accounts, tag):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    hist = {}; n = len(accounts); done=[0]; lk=threading.Lock()
    def work(a):
        txs = es.txlist(a, max_pages=MAX_PAGES, offset=OFFSET)
        with lk:
            hist[a]=txs; done[0]+=1
            if done[0] % 50 == 0: print(f"[{tag} {done[0]}/{n}] {len(txs)} txs", flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(work, accounts))
    return hist

def main():
    audit = pd.read_csv(os.path.join(DATA, "step1_audit.csv"))
    clean = audit[(audit.clean == True) & (audit.scammer.notna())]
    pos_accounts = sorted(clean.scammer.str.lower().unique().tolist())
    print(f"Clean phishing txs: {len(clean)}  unique scammer accounts: {len(pos_accounts)}")

    # benign senders
    ben = pd.read_csv(os.path.join(DATA, "ptx_benign_raw.csv"))
    ben = ben.sample(frac=1.0, random_state=42).reset_index(drop=True)
    from concurrent.futures import ThreadPoolExecutor
    cand = ben.value.tolist()[: MAX_BENIGN*4]
    txs = list(ThreadPoolExecutor(max_workers=10).map(es.get_tx, cand))
    ben_senders = []; seen = set()
    for tx in txs:
        if not tx: continue
        frm = (tx.get("from") or "").lower()
        if frm and frm not in seen:
            seen.add(frm); ben_senders.append(frm)
        if len(ben_senders) >= MAX_BENIGN: break
    print(f"Benign unique senders collected: {len(ben_senders)}", flush=True)

    pos_hist = crawl_accounts(pos_accounts, "POS")
    neg_hist = crawl_accounts(ben_senders, "NEG")

    with open(os.path.join(DATA, "step2_pos_hist.json"), "w") as f:
        json.dump(pos_hist, f)
    with open(os.path.join(DATA, "step2_neg_hist.json"), "w") as f:
        json.dump(neg_hist, f)
    # mapping account -> gt tx hashes (for transaction-level eval)
    gt = clean.groupby(clean.scammer.str.lower())["txhash"].apply(list).to_dict()
    with open(os.path.join(DATA, "step2_gt_map.json"), "w") as f:
        json.dump(gt, f)
    print("Saved histories. pos:", len(pos_hist), "neg:", len(neg_hist))

if __name__ == "__main__":
    main()
