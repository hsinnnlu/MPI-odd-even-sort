#!/usr/bin/env python3
"""讀取課程測資的 NN.txt，印出 "n nodes procs partition rounds"（空白分隔）。

不確定 NN.txt 的確切格式，所以同時支援：
  - JSON：{"n": 100, "nodes": 1, "procs": 4, ...}
  - 每行 key: value 或 key = value
找不到的欄位用預設值：nodes=1, procs=1, partition=big, rounds=25
"""
import json
import re
import sys

text = open(sys.argv[1]).read()
fields = {}
try:
    fields = {str(k).lower(): v for k, v in json.loads(text).items()}
except Exception:
    for m in re.finditer(r"([A-Za-z_]+)\s*[:=]\s*\"?([^\s\",}]+)", text):
        fields[m.group(1).lower()] = m.group(2)


def pick(names, default=None):
    for name in names:
        if name in fields:
            return fields[name]
    return default


n = pick(["n", "size", "num", "elements"])
if n is None:
    sys.exit(f"cannot find N in {sys.argv[1]}; content:\n{text}")
nodes = pick(["nodes", "node", "num_nodes"], 1)
procs = pick(["procs", "nproc", "np", "processes", "ntasks", "proc"], 1)
partition = pick(["partition", "part", "p"], "big")
rounds = pick(["hash_rounds", "rounds", "round", "hash_round"], 25)
print(n, nodes, procs, partition, rounds)
