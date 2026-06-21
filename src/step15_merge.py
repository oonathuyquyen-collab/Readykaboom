#!/usr/bin/env python3
"""STEP 15 - Merge Seaport GT into the audit, report new coverage, write step1_audit_v2.csv."""
import os, csv, json
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def main():
    gt = json.load(open(os.path.join(DATA, "seaport_gt.json")))
    by_tx = {g["txhash"]: g for g in gt}
    rows = list(csv.DictReader(open(os.path.join(DATA, "step1_audit.csv"))))
    fields = list(rows[0].keys())
    before = sum(1 for r in rows if r["clean"] == "True")
    added = 0
    for r in rows:
        if r["scam_type"] == "nft_order" and r["clean"] != "True" and r["txhash"] in by_tx:
            g = by_tx[r["txhash"]]
            r["clean"] = "True"
            r["scammer"] = g["scammer"]
            r["reason"] = "seaport_" + g["method"]
            added += 1
    after = sum(1 for r in rows if r["clean"] == "True")
    with open(os.path.join(DATA, "step1_audit_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    tot = len(rows)
    print(f"total txs           : {tot}")
    print(f"clean GT before     : {before} ({before/tot*100:.1f}%)")
    print(f"nft_order resolved  : {added}")
    print(f"clean GT after      : {after} ({after/tot*100:.1f}%)")
    # per scam_type after
    from collections import Counter
    cov = Counter(); tt = Counter()
    for r in rows:
        tt[r["scam_type"]] += 1
        if r["clean"] == "True": cov[r["scam_type"]] += 1
    print("coverage by scam_type (after):")
    for k in tt: print(f"  {k:18s} {cov[k]}/{tt[k]} ({cov[k]/tt[k]*100:.1f}%)")
    # distinct nft_order scammers
    nft_scammers = set(by_tx[r["txhash"]]["scammer"] for r in rows
                       if r["scam_type"] == "nft_order" and r["txhash"] in by_tx)
    print(f"distinct nft_order scammers: {len(nft_scammers)}")
    json.dump(sorted(nft_scammers), open(os.path.join(DATA, "nft_scammers.json"), "w"))

if __name__ == "__main__":
    main()
