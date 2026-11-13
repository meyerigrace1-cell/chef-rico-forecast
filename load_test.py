"""
Simple load test for the /predict endpoint (Milestone 5 - reliability under load).
Uses only the Python standard library.

Run:  python load_test.py                      (tests http://127.0.0.1:8000)
      python load_test.py https://YOUR-APP.onrender.com 200 20
      arguments: base_url  total_requests  concurrent_users
"""
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
total = int(sys.argv[2]) if len(sys.argv) > 2 else 100
users = int(sys.argv[3]) if len(sys.argv) > 3 else 10

payload = json.dumps({"date": "2026-11-13", "is_event": 1, "is_ntsa_crackdown": 0, "rain_mm": 0}).encode()


def one_call(_):
    req = urllib.request.Request(f"{base}/predict", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    return ok, (time.perf_counter() - start) * 1000


print(f"Sending {total} requests with {users} concurrent users to {base}/predict ...")
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=users) as pool:
    results = list(pool.map(one_call, range(total)))
elapsed = time.perf_counter() - t0

lat = sorted(ms for ok, ms in results if ok)
fails = sum(1 for ok, _ in results if not ok)
print(f"\nCompleted in {elapsed:.1f}s  ->  {total / elapsed:.1f} requests/second")
print(f"Successful: {len(lat)}   Failed: {fails}")
if lat:
    print(f"Latency  average {statistics.mean(lat):.0f} ms | median {statistics.median(lat):.0f} ms | "
          f"95th percentile {lat[int(len(lat) * 0.95) - 1]:.0f} ms | max {max(lat):.0f} ms")
