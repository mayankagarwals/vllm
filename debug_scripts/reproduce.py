#!/usr/bin/env python
"""drift-k7 最小复现运行器：单轴贪心约简（S0-S4 及 S5+ 自定义）。

每例：forced-reject（synthetic rates=[0]*K）vs target-only，3 次重复，
tau=0.5。采集双方完整 token IDs、首分歧位置 + 双侧 margin、分歧模式
（单点翻转 vs 分歧后走散）、manifest（含 BOOT uuid）。

场景构造：直接复用 specsafe.generator.matrix_drift35150() 的 k7 原始
场景做轴约简——S0/S1 保留 r006 原始 prompt（可比性）；S4 取 r006 单请求。

用法：
  python drift_min_run.py --case S4            # 网格内
  python drift_min_run.py --case S5 --k 7 --n 1 --len 128   # 自定义
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid

sys.path.insert(0, "/root/specsafe")

from specsafe.engine.vllm_offline import VllmEngine          # noqa: E402
from specsafe.generator import build_matrix                  # noqa: E402
from specsafe.invariants import (                            # noqa: E402
    _compare,
    _first_divergence,
    _margin_at,
    _verdict_of,
    run_arm,
    JudgeConfig,
)
from specsafe.scenario import EngineOptions, ReqSpec, Scenario  # noqa: E402

GRID = {
    # case: (K, n, L, request_selector)
    "S0": (7, 32, 256, "all"),
    "S1": (7, 8, 256, "first8"),      # r000-r007，含 r006
    "S2": (7, 32, 128, "all"),
    "S3": (5, 32, 256, "all"),
    "S4": (7, 1, 256, "r006"),        # 判别步：r006 单请求
}
TAU = 0.5
REPEATS = 3


def base_requests() -> list[ReqSpec]:
    """k7 原始 32 请求（与 drift35150 矩阵完全一致，确定性种子）。"""
    case = next(c for c in build_matrix("drift35150")
                if c.case_id == "drift-k7-n32-L256")
    return list(case.scenario.requests)


def make_case(tag: str, k: int, n: int, length: int, selector: str) -> Scenario:
    reqs = base_requests()
    if selector == "r006":
        chosen = [r for r in reqs if r.rid == "r006"]
    elif selector == "first8":
        chosen = reqs[:8]
    else:
        chosen = reqs[:n]
    chosen = [ReqSpec(r.rid, r.prompt, length) for r in chosen]
    opts = EngineOptions(
        method="spec", num_speculative_tokens=k, seed=k,
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=(0.0,) * k)
    return Scenario(f"drift-min/{tag}", f"{tag}: K={k} n={len(chosen)} L={length}",
                    tuple(chosen), opts, (), k)


def pattern_stats(a, b, div):
    """分歧模式：首分歧后 min 长度窗口内差异占比（高=走散，低=单点）。"""
    tail_a, tail_b = a[div:], b[div:]
    if not tail_a:
        return {"after_div_positions": 0, "diff_after": 0, "diff_ratio": None}
    diff = sum(1 for x, y in zip(tail_a, tail_b) if x != y)
    return {"after_div_positions": len(tail_a), "diff_after": diff,
            "diff_ratio": round(diff / len(tail_a), 3)}


def run_case(tag: str, sc: Scenario, model: str, out_dir: str) -> dict:
    boot = uuid.uuid4().hex[:8]
    judge = JudgeConfig(tau=TAU, repeats=1)   # 手动控制重复以采集每轮数据
    eng = VllmEngine(model=model, max_model_len=4096, max_num_seqs=64)
    t0 = time.time()
    repeats = []
    for rep in range(REPEATS):
        spec = run_arm(eng, sc, sc.spec_opts, label="spec_forced_reject")
        base = run_arm(eng, sc, EngineOptions(method="none", seed=sc.seed),
                       label="target_only")
        ms = _compare("spec_forced_reject", spec, "target_only", base,
                      False, TAU)
        rep_data = {"verdict": _verdict_of(ms), "mismatches": []}
        for m in ms:
            ra, rb = spec[m.rid], base[m.rid]
            d = {"rid": m.rid, "pos": m.first_divergence,
                 "margin_a": m.margin_a, "margin_b": m.margin_b,
                 "cls": m.verdict,
                 "pattern": pattern_stats(ra.token_ids, rb.token_ids,
                                          m.first_divergence
                                          if m.first_divergence is not None
                                          else min(len(ra.token_ids),
                                                   len(rb.token_ids)))}
            rep_data["mismatches"].append(d)
        rep_data["fail_class"] = [d for d in rep_data["mismatches"]
                                  if d["cls"] == "fail"]
        # 完整 token（双方）——只存分歧请求，节省体积
        rep_data["tokens"] = {
            m["rid"]: {"spec": spec[m["rid"]].token_ids,
                       "target": base[m["rid"]].token_ids}
            for m in rep_data["mismatches"]}
        repeats.append(rep_data)
        print(f"[{tag} rep{rep + 1}] {rep_data['verdict']} "
              f"(fail-class x{len(rep_data['fail_class'])})", flush=True)
    eng.shutdown()
    verdicts = [r["verdict"] for r in repeats]
    final = ("fail" if verdicts.count("fail") * 2 > len(verdicts)
             else "boundary" if sum(v != "pass" for v in verdicts) * 2 > len(verdicts)
             else "pass")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    report = {
        "case": tag, "K": sc.spec_opts.num_speculative_tokens,
        "n": len(sc.requests), "L": sc.requests[0].max_tokens,
        "boot": boot, "repeats": verdicts, "final": final,
        "tau": TAU, "duration_s": round(time.time() - t0, 1),
        "env": {"vllm": eng.info().details.get("vllm_version"),
                "model": model, "gpu": gpu, "ts": time.strftime("%F %T")},
    }
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{tag}__K{report['K']}_n{report['n']}_L{report['L']}__boot{boot}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "repeat_details": repeats}, f, indent=1)
    print(f"[{tag}] FINAL={final} repeats={verdicts} -> {path}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True,
                   help="S0-S4 或自定义 tag（配合 --k/--n/--len）")
    p.add_argument("--k", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--len", dest="length", type=int)
    p.add_argument("--selector", default="all", help="all | first8 | r006")
    p.add_argument("--model", default="/root/autodl-tmp/models/qwen3.8-27b-fp8")
    p.add_argument("--out", default="/root/specsafe/results/drift_min")
    a = p.parse_args()

    if a.case in GRID:
        k, n, L, sel = GRID[a.case]
        k, n, L = (a.k or k), (a.n or n), (a.length or L)
    else:
        k, n, L = a.k, a.n, a.length
        assert all(v is not None for v in (k, n, L)), "--k/--n/--len required"
        sel = a.selector
    sc = make_case(a.case, k, n, L, sel)
    run_case(a.case, sc, a.model, a.out)


if __name__ == "__main__":
    main()
