"""Cached Etherscan V2 API client (chainid=1, Ethereum mainnet).
SQLite cache so re-runs are free and resumable. Conservative rate limiting."""
import os, time, json, sqlite3, threading
import requests

API = "https://api.etherscan.io/v2/api"
CHAINID = 1
KEY = os.environ.get("ETHERSCAN_KEY", "QQD2RT4RGBVCCIJFH1ETZZWBJR55AU1YYV")
CACHE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "etherscan_cache.sqlite")

_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 0.21

def _conn():
    os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)
    c = sqlite3.connect(CACHE_DB, timeout=60)
    c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
    return c

def _cache_get(k):
    c = _conn()
    try:
        r = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None
    finally:
        c.close()

def _cache_put(k, v):
    c = _conn()
    try:
        c.execute("INSERT OR REPLACE INTO cache (k,v) VALUES (?,?)", (k, json.dumps(v)))
        c.commit()
    finally:
        c.close()

def _throttle():
    with _lock:
        dt = time.time() - _last[0]
        if dt < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - dt)
        _last[0] = time.time()

def _request(params, retries=5):
    params = dict(params); params["chainid"] = CHAINID; params["apikey"] = KEY
    j = {}
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.get(API, params=params, timeout=40)
            j = r.json()
        except Exception:
            time.sleep(1.5 * (attempt + 1)); continue
        msg = str(j.get("message", "")); res = j.get("result", "")
        if "rate limit" in str(res).lower() or "rate limit" in msg.lower() or "Max calls" in msg:
            time.sleep(1.0 * (attempt + 1)); continue
        return j
    return j

def get_tx(txhash):
    k = f"tx::{txhash}"
    v = _cache_get(k)
    if v is not None: return v
    j = _request({"module":"proxy","action":"eth_getTransactionByHash","txhash":txhash})
    res = j.get("result"); _cache_put(k, res); return res

def get_code(address):
    address = address.lower(); k = f"code::{address}"
    v = _cache_get(k)
    if v is not None: return v
    j = _request({"module":"proxy","action":"eth_getCode","address":address,"tag":"latest"})
    res = j.get("result","0x"); _cache_put(k, res); return res

def is_contract(address):
    if not address: return None
    return len(get_code(address)) > 2

def txlist(address, max_pages=1, offset=10000):
    address = address.lower(); out = []
    for page in range(1, max_pages+1):
        k = f"txlist::{address}::{page}::{offset}"
        v = _cache_get(k)
        if v is None:
            j = _request({"module":"account","action":"txlist","address":address,
                          "startblock":0,"endblock":99999999,"page":page,"offset":offset,"sort":"asc"})
            v = j.get("result",[]) if isinstance(j.get("result"), list) else []
            _cache_put(k, v)
        out.extend(v)
        if len(v) < offset: break
    return out

def tokentx(address, max_pages=1, offset=10000):
    address = address.lower(); out = []
    for page in range(1, max_pages+1):
        k = f"tokentx::{address}::{page}::{offset}"
        v = _cache_get(k)
        if v is None:
            j = _request({"module":"account","action":"tokentx","address":address,
                          "startblock":0,"endblock":99999999,"page":page,"offset":offset,"sort":"asc"})
            v = j.get("result",[]) if isinstance(j.get("result"), list) else []
            _cache_put(k, v)
        out.extend(v)
        if len(v) < offset: break
    return out

def get_receipt(txhash):
    k = f"receipt::{txhash}"
    v = _cache_get(k)
    if v is not None: return v
    j = _request({"module":"proxy","action":"eth_getTransactionReceipt","txhash":txhash})
    res = j.get("result"); _cache_put(k, res); return res

def get_internal_tx(txhash):
    k = f"internal::{txhash}"
    v = _cache_get(k)
    if v is not None: return v
    j = _request({"module":"account","action":"txlistinternal","txhash":txhash})
    res = j.get("result",[]) if isinstance(j.get("result"), list) else []
    _cache_put(k, res); return res
