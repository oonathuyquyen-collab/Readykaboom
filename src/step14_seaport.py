#!/usr/bin/env python3
"""STEP 14 - Resolve GT scammer cashier for nft_order txs via NET on-chain flow in the receipt.
Net flow = value/NFTs received minus sent, per address; scammer = max-net EOA recipient
(this isolates the final cashier from marketplace/conduit pass-through addresses).
Covers paid orders (ERC20/WETH+ETH) and zero-price NFT theft (ERC721). Excludes upgradeTo
infra txs (no transfers) which are reported separately as non-money infra.
"""
import sys, os, csv, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etherscan as es

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
ZERO = "0x0000000000000000000000000000000000000000"

def ta(t): return "0x" + t[-40:].lower()

def pick_max_net(net, excl, contract_cache):
    cands = sorted(((a, v) for a, v in net.items() if v > 0 and a not in excl), key=lambda x: -x[1])
    for addr, v in cands:
        if addr not in contract_cache:
            try: contract_cache[addr] = es.is_contract(addr)
            except Exception: contract_cache[addr] = False
        if not contract_cache[addr]:
            return addr, v, True
    return (cands[0][0], cands[0][1], False) if cands else (None, 0, False)

def resolve(txhash, submitter, to_market, contract_cache):
    rec = es.get_receipt(txhash)
    if not rec: return None
    logs = rec.get("logs", []) or []
    val_net = {}; nft_net = {}
    for lg in logs:
        topics = lg.get("topics", []) or []
        if not topics or topics[0].lower() != TRANSFER: continue
        if len(topics) == 3:                       # ERC20/WETH value transfer
            try: v = int(lg.get("data", "0x"), 16)
            except Exception: v = 0
            if v <= 0: continue
            frm = ta(topics[1]); to = ta(topics[2])
            val_net[to] = val_net.get(to, 0) + v
            val_net[frm] = val_net.get(frm, 0) - v
        elif len(topics) == 4:                     # ERC721 transfer (count units)
            frm = ta(topics[1]); to = ta(topics[2])
            nft_net[to] = nft_net.get(to, 0) + 1
            nft_net[frm] = nft_net.get(frm, 0) - 1
    excl = {submitter, to_market, ZERO, WETH}
    # internal ETH
    try:
        for it in es.get_internal_tx(txhash):
            to = (it.get("to") or "").lower()
            try: v = int(it.get("value", "0"))
            except Exception: v = 0
            if to and v > 0: val_net[to] = val_net.get(to, 0) + v
    except Exception: pass
    # prefer monetary inflow
    addr, v, is_eoa = pick_max_net(val_net, excl, contract_cache)
    if addr:
        return {"txhash": txhash, "scammer": addr, "value_eth": v / 1e18,
                "method": "net_value" + ("" if is_eoa else "_contract")}
    addr, c, is_eoa = pick_max_net(nft_net, excl, contract_cache)
    if addr:
        return {"txhash": txhash, "scammer": addr, "value_eth": 0.0,
                "method": "net_nft" + ("" if is_eoa else "_contract"), "nft_count": c}
    return None

def main():
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "step1_audit.csv")))
            if r["scam_type"] == "nft_order"]
    print(f"nft_order rows: {len(rows)}", flush=True)
    cc = {}; out = []; infra = 0; t0 = time.time()
    for i, r in enumerate(rows):
        sub = (r.get("from") or "").lower(); mk = (r.get("to") or "").lower()
        if r.get("subtype") == "proxy_upgrade":
            infra += 1; continue
        try: res = resolve(r["txhash"], sub, mk, cc)
        except Exception: res = None
        if res: res["subtype"] = r.get("subtype"); out.append(res)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] resolved={len(out)} infra={infra} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(DATA, "seaport_gt.json"), "w"), indent=1)
    print(f"DONE resolved {len(out)} (infra/proxy_upgrade skipped {infra}) "
          f"of {len(rows)} -> seaport_gt.json", flush=True)

if __name__ == "__main__":
    main()
