"""STEP 1 - Feasibility audit (gating experiment).
For every PTXPhish phishing tx: fetch from/to/value/selector, resolve which side is the
scammer, and decide whether the tx is a CLEAN transaction-level GT instance (i.e. it appears
directly in the scammer account's external history).

Scammer-side rules grounded in the per-category samples we inspected:
  * payable_function (airdrop/wallet): victim -> phishing contract, value>0 in ETH.
      scammer endpoint = `to` (contract). tx is INBOUND to it. CLEAN.
  * address_poisoning (zero/dust/fake): scammer -> victim spam transfer.
      scammer = `from` (EOA). tx is OUTBOUND from it. CLEAN.
  * ice_phishing (approve/permit/setapprovalforall): the recorded tx is the malicious
      transferFrom/safeTransferFrom executed BY the scammer (caller). scammer = `from` (EOA).
      tx OUTBOUND. CLEAN when `from` is an EOA (caller-is-scammer).
  * nft_order (seaport/conduit): executed via a marketplace contract; scammer not a direct
      from/to party -> NOT clean (needs log tracing -> future work).
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import etherscan as es

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 4-byte selectors -> human label
SELECTORS = {
    "0x095ea7b3": "approve", "0x23b872dd": "transferFrom", "0xa9059cbb": "transfer",
    "0x42842e0e": "safeTransferFrom", "0x23b872dd": "transferFrom", "0xa22cb465": "setApprovalForAll",
    "0xcaa5c23f": "permit_drainer", "0xd505accf": "permit", "0x3659cfe6": "upgradeTo",
    "0x32389b71": "bulk_transfer", "0xb3be57f8": "buy_order", "0x3158952e": "airdrop_fn",
    "0x5fba79f5": "wallet_fn", "0x": "eth_transfer",
}

def classify(sub, frm, to, val_eth, sel, from_code, to_code):
    """return (scammer_addr, direction, clean_bool, reason)"""
    if sub in ("airdrop", "wallet"):
        # payable function: money flows victim -> scammer endpoint (`to`)
        if to and val_eth > 0:
            return to, "IN", True, "payable_to"
        return (to or frm), "IN", bool(to), "payable_to_noval"
    if sub in ("zero_value", "fake_token", "dust_value"):
        # poisoning: scammer initiates spam -> from is scammer (EOA)
        if from_code is False:
            return frm, "OUT", True, "poison_from_eoa"
        return frm, "OUT", False, "poison_from_contract"
    if sub in ("approve", "permit", "setapprovalforall"):
        # recorded tx is the scammer-executed drain (caller=scammer)
        if from_code is False:
            return frm, "OUT", True, "drain_from_eoa"
        return frm, "OUT", False, "drain_from_contract"
    if sub in ("bulk_transfer", "proxy_upgrade", "free_buy_order"):
        # marketplace/proxy intermediary; scammer not a direct party
        return None, None, False, "marketplace_intermediary"
    return None, None, False, "unknown"

def main():
    ph = pd.read_csv(os.path.join(DATA, "ptx_phishing_raw.csv"))
    n = len(ph)
    from concurrent.futures import ThreadPoolExecutor
    import threading
    done = [0]; lk = threading.Lock()
    def work(rec):
        i, r = rec
        h = r["value"]
        tx = es.get_tx(h)
        with lk:
            done[0]+=1
            if done[0] % 200 == 0: print(f"[{done[0]}/{n}] processed", flush=True)
        if not tx:
            return {"txhash":h,"branch":r.branch,"scam_type":r.scam_type,"subtype":r.subtype,"status":"tx_not_found"}
        frm = (tx.get("from") or "").lower()
        to = (tx.get("to") or "").lower() or None
        val = int(tx.get("value","0x0"), 16)/1e18
        sel = (tx.get("input","0x")[:10]) if tx.get("input","0x") != "0x" else "0x"
        from_code = es.is_contract(frm) if frm else None
        to_code = es.is_contract(to) if to else None
        scammer, direction, clean, reason = classify(r.subtype, frm, to, val, sel, from_code, to_code)
        return {
            "txhash": h, "branch": r.branch, "scam_type": r.scam_type, "subtype": r.subtype,
            "from": frm, "to": to, "value_eth": val, "selector": sel,
            "sel_name": SELECTORS.get(sel, "other"),
            "from_is_contract": from_code, "to_is_contract": to_code,
            "scammer": scammer, "direction": direction, "clean": clean, "reason": reason,
            "block": int(tx.get("blockNumber","0x0"),16),
            "ts": int(tx.get("blockTimestamp","0x0"),16) if tx.get("blockTimestamp") else None,
            "status": "ok",
        }
    with ThreadPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(work, list(ph.iterrows())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA, "step1_audit.csv"), index=False)
    print("\nSAVED step1_audit.csv  rows:", len(out))
    print(out.groupby(["scam_type","subtype","clean"]).size())
    print("\nclean per branch:")
    print(out.groupby(["branch","clean"]).size())

if __name__ == "__main__":
    main()
