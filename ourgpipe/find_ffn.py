import json, glob, os

base = "./profiler_thread-stream_4096"

for path in glob.glob(os.path.join(base, "rank*", "*.pt.trace.json")):
    with open(path, "r") as f:
        data = json.load(f)
    names = {e.get("name", "") for e in data.get("traceEvents", [])}
    hits = [n for n in names if "ffn" in n.lower()]
    print(path, " -> ffn-like events:", hits)
