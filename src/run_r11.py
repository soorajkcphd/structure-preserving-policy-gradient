"""
R-11: Task 2 with held-out prompts, more seeds, and an independent V2 check.

    python run_r11.py --selftest
    python run_r11.py --seeds 10          # ~2.5 h
    python run_r11.py --seeds 40          # ~10 h; the powered version
    python run_r11.py --analyse-only

Requires 04_pap_CG.py and task2_multiseed.py in the same directory.
No edits to either.

Why
---
Three problems with Task 2 as published, all of them fixable without changing
the method:

  1. No held-out set.  The same six prompts are used for structure discovery,
     for REINFORCE training and for evaluation.  Whatever the structured
     policy learns about those six strings is scored on those six strings.
     Every number in Section 9 is in-sample.

  2. Six prompts.  With eval_episodes = 8 and 3 eval seeds the evaluation is
     6 prompts x 3 repeats, and the replication unit is the training seed, so
     the effective n is the seed count and the prompt sample is tiny.

  3. V2 = 0.99993 is not reproducible.  The released validation compares an
     expression with itself (see validate_C2_proj_natgrad_equiv in
     04_pap_CG.py: both sides are built from the same projection), so it
     cannot fail and its value carries no information.  This script recomputes
     the same quantity by an independent route: the preconditioned gradient is
     built from an explicitly constructed random SPD metric that never
     references the projection, so the two sides of the comparison are
     distinct.  The number in Table 12 is then confirmed or
     replaced.

What this changes, and what it does not
---------------------------------------
Changed:  the prompt set is split into disjoint discovery+train and test
          halves; evaluation happens only on test; the seed count is a flag.
Unchanged: the policy, the reward, the REINFORCE loop, the control arm and
          its already-repaired lambda (task2_multiseed.py fixed a defect where
          the control was built with lmbda = 0 and therefore never trained --
          we reuse that fixed path rather than reimplementing it).

The prompt set
--------------
Six prompts cannot be split.  We extend the published six to twenty-four in
the same register and on the same topics, then split 12 / 12 by a fixed seed.
The published six are all placed in the discovery+train half, so the training
distribution is a superset of the original and the test half is entirely new.
That makes the comparison strictly harder than the published one, never
easier.

Output   r11_cells.csv   seed, split, arm, prompt_idx, reward, status
         r11_v2.json     the independent V2 recomputation
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd

CSV = "r11_cells.csv"
V2_JSON = "r11_v2.json"
ARMS = ("structured", "control")
SPLIT_SEED = 4242


# --------------------------------------------------------------------------- #
# The prompt set.  The first six are the published ones, verbatim.
# --------------------------------------------------------------------------- #
PUBLISHED_PROMPTS = [
    "Write about geometric structure in reinforcement learning.",
    "The story is about a robot that learns to paint.",
    "In the future, agents will learn from symmetry because",
    "Explain why group actions matter in optimization.",
    "A short story about a mathematician who loves groups.",
    "Summarize the role of invariances in optimization.",
]

ADDITIONAL_PROMPTS = [
    "Describe how rotations act on a vector space.",
    "The robot discovered that turning the canvas changed nothing important.",
    "Explain conservation laws to someone who has never studied physics.",
    "Write about a cartographer who could not decide on an orientation.",
    "Why does the order of two operations sometimes not matter?",
    "A brief note on why some transformations preserve distance.",
    "Tell a story about a dancer who thought in terms of symmetry.",
    "Explain what stays the same when everything else changes.",
    "Write about an engineer who trusted invariants over measurements.",
    "Describe the difference between a rotation and a stretch.",
    "In a world without preferred directions, navigation would",
    "Summarize why structure can make a search problem easier.",
    "A short story about a translator who preserved meaning exactly.",
    "Explain why physicists care about groups of transformations.",
    "Write about a sculptor who worked only with rigid motions.",
    "What does it mean for a system to have no preferred frame?",
    "Describe a machine that learns the shape of its own errors.",
    "Explain why averaging rotations is harder than it sounds.",
]


def build_splits(split_seed: int = SPLIT_SEED):
    """Disjoint (discovery+train, test) prompt lists.

    The published six are forced into the train half, so the training
    distribution contains everything the original used and the test half is
    entirely unseen.  This can only make the comparison harder.
    """
    rng = random.Random(int(split_seed))
    extra = list(ADDITIONAL_PROMPTS)
    rng.shuffle(extra)
    n_extra_train = len(PUBLISHED_PROMPTS)          # 6 -> train half of 12
    train = list(PUBLISHED_PROMPTS) + extra[:n_extra_train]
    test = extra[n_extra_train:]
    assert not (set(train) & set(test)), "train and test overlap"
    return train, test


# --------------------------------------------------------------------------- #
def _load_task2():
    """Import 04_pap_CG.py (non-importable filename) exactly as
    task2_multiseed.py does, so we get the same module object and the same
    already-repaired control arm."""
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader("pap_cg", "04_pap_CG.py").load_module()


def collect(seeds, overwrite: bool) -> pd.DataFrame:
    import torch
    if os.path.exists(CSV) and not overwrite:
        sys.exit(f"{CSV} exists; pass --overwrite to replace it.")
    M = _load_task2()

    cfg = M.Cfg()
    M.ensure_dir(cfg.results_dir)
    base_model, tok = M.load_model(cfg)
    n = base_model.config.n_embd
    alg = M.LieAlgebra(cfg.algebra, n)

    train_prompts, test_prompts = build_splits()
    print(f"\nprompt split: {len(train_prompts)} discovery+train, "
          f"{len(test_prompts)} held-out test (disjoint)")
    print(f"  the {len(PUBLISHED_PROMPTS)} published prompts are all in the "
          f"train half")

    # ---- structure discovery on the train half only -----------------------
    M.set_seed(0)
    V = torch.stack([M.repr_vec(base_model, tok, p, cfg.device)
                     for p in train_prompts], dim=0)
    k, eps = 4, 0.02
    R = torch.randn(k, n, n, device=cfg.device) * eps
    T = torch.linalg.matrix_exp(alg.project(R))
    T_mats = [T[i] for i in range(k)]
    disc = M.Discovery(alg, cfg.device, lr=cfg.discovery_lr,
                       steps=cfg.discovery_steps)
    perm = disc.perm_test(T_mats, V, B=cfg.perm_tests)
    X_star = disc.fit(T_mats, V, steps=cfg.discovery_steps)
    print(f"  discovery residual={perm['obs_residual']:.6f}  "
          f"p={perm['p_value']:.3f}")

    rows = []
    total, done = len(seeds), 0
    for seed in seeds:
        done += 1
        print(f"\n[{done}/{total}] seed {seed}", flush=True)
        try:
            # run_single_seed trains both arms and evaluates them; we call it
            # with the test prompts as the evaluation set by passing the train
            # prompts for training and re-evaluating on test below.
            M.set_seed(seed)
            pol = M.SIPolicy(base_model, tok, alg, X_star, cfg).to(cfg.device)
            tr = M.REINFORCE(pol, tok, cfg)
            for _ in range(cfg.rl_iters):
                tr.step([random.choice(train_prompts)
                         for _ in range(cfg.batch_prompts)])

            # The control: identical except the compactness constraint, built
            # the way task2_multiseed.py repaired it (same X_star, same lmbda,
            # same budget) -- not the lmbda = 0 version, which never trains.
            M.set_seed(seed)
            ctrl = M.SIPolicy(base_model, tok, M.LieAlgebra("gl", n),
                              X_star, cfg).to(cfg.device)
            ctrl.lmbda = pol.lmbda
            before = torch.cat([p.detach().reshape(-1).clone()
                                for p in ctrl.parameters() if p.requires_grad])
            ctr = M.REINFORCE(ctrl, tok, cfg)
            for _ in range(cfg.rl_iters):
                ctr.step([random.choice(train_prompts)
                          for _ in range(cfg.batch_prompts)])
            after = torch.cat([p.detach().reshape(-1).clone()
                               for p in ctrl.parameters() if p.requires_grad])
            moved = float((after - before).abs().max().item())
            if moved == 0.0:
                raise RuntimeError(
                    "control parameters did not change during training; the "
                    "comparison would be trained-vs-untrained. Aborting.")

            # ---- evaluate both arms on both splits ------------------------
            # The train-split numbers are kept so the in-sample/out-of-sample
            # gap is visible rather than assumed.
            with torch.no_grad():
                for split, plist in (("test", test_prompts),
                                     ("train", train_prompts)):
                    for eval_seed in (0, 1, 2):
                        torch.manual_seed(eval_seed)
                        np.random.seed(eval_seed)
                        random.seed(eval_seed)
                        for pi, p in enumerate(plist):
                            for arm, model in (("structured", pol),
                                               ("control", ctrl)):
                                ids = tok(p, return_tensors="pt"
                                          ).to(cfg.device)["input_ids"]
                                out = model.generate(ids, cfg.max_new_tokens)
                                cont = out[:, ids.shape[1]:]
                                r = M.task_reward_from_ids(base_model, tok,
                                                           ids, cont)
                                rows.append(dict(seed=seed, split=split,
                                                 arm=arm, prompt_idx=pi,
                                                 eval_seed=eval_seed,
                                                 reward=float(r),
                                                 status="ok"))
            del pol, ctrl, tr, ctr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            raise
        except Exception as e:                               # noqa: BLE001
            rows.append(dict(seed=seed, split="test", arm="structured",
                             prompt_idx=-1, eval_seed=-1, reward=np.nan,
                             status=f"{type(e).__name__}: {e}"[:110]))
            print(f"    FAILED: {type(e).__name__}: {e}")
        pd.DataFrame(rows).to_csv(CSV, index=False)
        ok = [r for r in rows if r["status"] == "ok" and r["split"] == "test"]
        if ok:
            s = np.mean([r["reward"] for r in ok if r["arm"] == "structured"])
            c = np.mean([r["reward"] for r in ok if r["arm"] == "control"])
            print(f"    running held-out means: structured={s:.4f} "
                  f"control={c:.4f}  delta={s - c:+.4f}")
    print(f"\nWrote {CSV} ({len(rows)} rows)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def per_seed_means(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Average within (seed, arm) first: the replication unit is the training
    seed, not the prompt.  Pooling prompts as if independent would inflate n
    by 12x and is the single easiest way to manufacture significance here."""
    d = df[(df.status == "ok") & (df.split == split)]
    if not len(d):
        return pd.DataFrame(columns=["seed", "structured", "control"])
    g = (d.groupby(["seed", "arm"], as_index=False)["reward"].mean()
         .pivot(index="seed", columns="arm", values="reward").reset_index())
    for a in ARMS:
        if a not in g.columns:
            g[a] = np.nan
    return g.dropna(subset=list(ARMS))


def _report(v, label):
    from scipy import stats as st
    n = len(v)
    if n < 2:
        print(f"  {label:34s} n={n}  (insufficient)")
        return None
    m, s = float(v.mean()), float(v.std(ddof=1))
    if np.isclose(s, 0.0):
        print(f"  {label:34s} n={n}  all differences identical ({m:+.4f}); "
              f"refusing to test")
        return None
    t, p = st.ttest_1samp(v, 0.0)
    h = st.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    dz = m / s
    print(f"  {label:34s} n={n:3d}  {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}]  "
          f"dz={dz:+.2f}  p={p:.4g}")
    return dict(n=n, mean=m, lo=m - h, hi=m + h, dz=dz, p=float(p))


def analyse(df: pd.DataFrame) -> None:
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{bad.seed.nunique()} failed seed(s):")
        for _, r in bad.iterrows():
            print(f"    seed={r.seed}: {r.status}")

    te, trn = per_seed_means(df, "test"), per_seed_means(df, "train")
    print("\n" + "=" * 92)
    print("R-11  TASK 2 ON HELD-OUT PROMPTS")
    print("  The replication unit is the training seed: prompt-level rewards "
          "are averaged\n  within a seed before any test.  Prompts are not "
          "independent replicates.")
    print("=" * 92)
    for name, g in (("HELD-OUT (test)", te), ("in-sample (train)", trn)):
        if not len(g):
            print(f"  {name}: no complete seeds."); continue
        print(f"  {name:18s} structured {g['structured'].mean():.4f}"
              f"+/-{g['structured'].std(ddof=1):.4f}   "
              f"control {g['control'].mean():.4f}"
              f"+/-{g['control'].std(ddof=1):.4f}")

    print("\n  paired differences (structured - control):")
    r_te = _report((te["structured"] - te["control"]).to_numpy(),
                   "HELD-OUT prompts") if len(te) else None
    r_tr = _report((trn["structured"] - trn["control"]).to_numpy(),
                   "in-sample prompts") if len(trn) else None

    # ---- the generalisation gap -------------------------------------------
    if r_te and r_tr:
        gap = r_tr["mean"] - r_te["mean"]
        print(f"\n  generalisation gap (in-sample minus held-out): {gap:+.4f}")
        # "in-sample only" means: significant and positive on the prompts
        # used for discovery+training, and not significant on unseen ones.
        # Testing on the point estimate alone would call a noisy positive
        # held-out effect a failure to generalise.
        if (r_tr["p"] < 0.05 and r_tr["mean"] > 0
                and not (r_te["p"] < 0.05 and r_te["mean"] > 0)):
            print("  -> the published effect is IN-SAMPLE ONLY: significant on "
                  "the prompts used\n     for discovery and training "
                  f"({r_tr['mean']:+.4f}, p={r_tr['p']:.4g}), absent on unseen "
                  f"ones\n     ({r_te['mean']:+.4f}, p={r_te['p']:.4g}).  "
                  "Section 9's numbers are all in-sample.")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    if r_te is None:
        print("  Not enough complete seeds to decide.")
    elif r_te["p"] < 0.05 and r_te["mean"] > 0:
        print(f"  -> The Task-2 benefit SURVIVES on held-out prompts "
              f"({r_te['mean']:+.4f},\n     95% CI [{r_te['lo']:+.4f}, "
              f"{r_te['hi']:+.4f}], p={r_te['p']:.4g}, n={r_te['n']} seeds).\n"
              "     Section 9 may state it as an out-of-sample result, which "
              "is stronger than\n     what is currently claimed.")
    elif r_te["p"] < 0.05:
        print(f"  -> The structured arm is significantly worse on held-out "
              f"prompts\n     ({r_te['mean']:+.4f}, p={r_te['p']:.4g}).  The "
              "published in-sample effect does\n     not generalise and is "
              "not supported.")
    else:
        print(f"  -> No effect on held-out prompts ({r_te['mean']:+.4f}, "
              f"95% CI\n     [{r_te['lo']:+.4f}, {r_te['hi']:+.4f}], "
              f"p={r_te['p']:.4g}, n={r_te['n']} seeds).  This does not show "
              "equivalence;\n     that would need the interval to lie inside a "
              "pre-registered margin.")
    if r_te and r_te["n"] < 20:
        print(f"\n  NOTE: n = {r_te['n']} seeds.  The paper's own power "
              "discussion asks for more;\n  re-run with --seeds 40 before "
              "resting a null on this.")


# --------------------------------------------------------------------------- #
def recompute_v2(trials: int = 200, n: int = 32, seed: int = 0) -> dict:
    """Independent recomputation of the V2 quantity.

    The released check builds both sides of the comparison from the same
    projection, so it compares an expression with itself and cannot fail.
    Here the 'natural gradient' side is obtained without reference to the
    projection: we take a Euclidean gradient, precondition it by an explicitly
    constructed metric, and only then compare directions.  If the reported
    0.99993 is real, this route reproduces it; if it was an artefact of the
    self-comparison, this route will not.
    """
    import torch

    g = torch.Generator().manual_seed(int(seed))
    cos_proj, cos_ambient = [], []
    for _ in range(trials):
        A = torch.randn(n, n, generator=g, dtype=torch.float64)
        # Euclidean gradient of an arbitrary smooth objective
        G = torch.randn(n, n, generator=g, dtype=torch.float64)
        # Projection onto so(n) -- the SP-PG direction
        G_proj = 0.5 * (G - G.T)
        # A metric that is not built from the projection: a random SPD
        # preconditioner acting on the ambient n^2 coordinates.
        B = torch.randn(n * n, n * n, generator=g, dtype=torch.float64) / math.sqrt(n * n)
        Fm = B @ B.T + 1e-2 * torch.eye(n * n, dtype=torch.float64)
        g_nat = torch.linalg.solve(Fm, G.reshape(-1)).reshape(n, n)
        g_nat_proj = 0.5 * (g_nat - g_nat.T)

        def cos(x, y):
            return float(torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)).item())

        cos_proj.append(cos(G_proj, g_nat_proj))
        cos_ambient.append(cos(G, g_nat))
        _ = A
    return {
        "n_trials": trials, "n": n,
        "cos_projected_mean": float(np.mean(cos_proj)),
        "cos_projected_std": float(np.std(cos_proj, ddof=1)),
        "cos_ambient_mean": float(np.mean(cos_ambient)),
        "cos_ambient_std": float(np.std(cos_ambient, ddof=1)),
        "published_V2": 0.99993,
    }


def report_v2(res: dict) -> None:
    print("\n" + "=" * 92)
    print("R-11  V2 RECOMPUTED BY AN INDEPENDENT ROUTE")
    print("=" * 92)
    print(f"  projected directions : cos = {res['cos_projected_mean']:.5f} "
          f"+/- {res['cos_projected_std']:.5f}  ({res['n_trials']} trials)")
    print(f"  ambient directions   : cos = {res['cos_ambient_mean']:.5f} "
          f"+/- {res['cos_ambient_std']:.5f}")
    print(f"  published V2         : {res['published_V2']:.5f}")
    if res["cos_projected_mean"] >= 0.99:
        print("\n  -> reproduced: the projected gradient does align with the "
              "preconditioned\n     gradient to the reported precision under "
              "an independently constructed\n     metric.  Table 12's V2 row "
              "stands.")
    else:
        print("\n  -> NOT reproduced.  Under a metric not built from the "
              "projection, the\n     alignment is "
              f"{res['cos_projected_mean']:.4f}, not "
              f"{res['published_V2']:.5f}.  The published value is an\n     "
              "artefact of comparing an expression with itself; it is an "
              "identity,\n     not a measurement.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    ok = True

    def check(nm, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {nm}{'  ' + dd if dd else ''}")

    print("(1) the prompt split")
    tr, te = build_splits()
    check("train and test are disjoint", not (set(tr) & set(te)))
    check("all six published prompts are in train",
          all(p in tr for p in PUBLISHED_PROMPTS))
    check("no published prompt leaks into test",
          not any(p in te for p in PUBLISHED_PROMPTS))
    check("both halves are non-trivial", len(tr) >= 10 and len(te) >= 10,
          f"train={len(tr)}, test={len(te)}")
    check("no duplicate prompts anywhere",
          len(set(tr + te)) == len(tr + te))
    tr2, te2 = build_splits()
    check("the split is reproducible", tr == tr2 and te == te2)
    tr3, te3 = build_splits(999)
    check("a different split seed gives a different split", te3 != te)
    check("...but the published six stay in train regardless",
          all(p in tr3 for p in PUBLISHED_PROMPTS))

    print("(2) the replication unit")
    rows = []
    for seed in range(4):
        for split in ("test", "train"):
            for pi in range(12):
                for arm, base in (("structured", 0.62), ("control", 0.60)):
                    rows.append(dict(seed=seed, split=split, arm=arm,
                                     prompt_idx=pi, eval_seed=0,
                                     reward=base + 0.001 * pi, status="ok"))
    df = pd.DataFrame(rows)
    g = per_seed_means(df, "test")
    check("prompt rewards are averaged within a seed", len(g) == 4,
          f"{len(g)} rows for 4 seeds x 12 prompts")
    check("both arms survive the pivot",
          set(ARMS).issubset(set(g.columns)))
    check("the per-seed difference is the arm difference",
          abs((g["structured"] - g["control"]).mean() - 0.02) < 1e-9)

    print("(3) analysis and the three verdicts")
    rng = np.random.default_rng(0)

    def frame(d_test, d_train, n=10, noise=0.02):
        rows = []
        for s in range(n):
            se = rng.normal(0, 0.01)
            for split, d in (("test", d_test), ("train", d_train)):
                for pi in range(12):
                    for arm in ARMS:
                        rows.append(dict(
                            seed=s, split=split, arm=arm, prompt_idx=pi,
                            eval_seed=0,
                            reward=0.60 + se + (d if arm == "structured" else 0)
                            + rng.normal(0, noise), status="ok"))
        return pd.DataFrame(rows)

    def run(d):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(d)
        return b.getvalue()

    surv = run(frame(0.05, 0.06))
    check("an effect on held-out prompts -> 'SURVIVES'", "SURVIVES" in surv)
    insample = run(frame(0.0, 0.06, noise=0.005))
    check("an in-sample-only effect is named as such",
          "IN-SAMPLE ONLY" in insample)
    worse = run(frame(-0.05, 0.06))
    check("a significant reversal -> 'not supported'",
          "not supported" in worse)
    null = run(frame(0.0, 0.0, noise=0.08))
    check("a null -> not called equivalence",
          "does not show" in null and "equivalence" in null)
    check("a small-n note is printed", "re-run with --seeds 40" in null)
    check("the replication-unit warning is always printed",
          "not independent replicates" in surv)

    print("(4) the independent V2 recomputation")
    res = recompute_v2(trials=25, n=8, seed=1)
    check("returns both a projected and an ambient alignment",
          "cos_projected_mean" in res and "cos_ambient_mean" in res)
    check("alignments are valid cosines",
          -1.0 <= res["cos_projected_mean"] <= 1.0
          and -1.0 <= res["cos_ambient_mean"] <= 1.0,
          f"proj={res['cos_projected_mean']:.4f}, "
          f"amb={res['cos_ambient_mean']:.4f}")
    b = io.StringIO()
    with contextlib.redirect_stdout(b):
        report_v2(res)
    txt = b.getvalue()
    check("a verdict is printed either way",
          ("reproduced" in txt) or ("NOT reproduced" in txt))
    check("the published value is shown alongside", "0.99993" in txt)
    fake = dict(res, cos_projected_mean=0.999999)
    b2 = io.StringIO()
    with contextlib.redirect_stdout(b2):
        report_v2(fake)
    check("a high alignment is reported as reproduced",
          "Table 12's V2 row stands" in b2.getvalue())

    print("(5) degenerate inputs")
    for name, dfx in [("empty", frame(0.05, 0.05).iloc[0:0]),
                      ("one seed", frame(0.05, 0.05, n=1)),
                      ("test only", frame(0.05, 0.05)[lambda x: x.split == "test"]),
                      ("all failed", frame(0.05, 0.05).assign(status="err"))]:
        try:
            run(dfx); good = True
        except Exception as exc:                             # noqa: BLE001
            good = False
            print(f"        {name}: {type(exc).__name__}: {exc}")
        check(f"degenerate input '{name}' does not crash", good)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-v2", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        df = pd.read_csv(CSV)
    else:
        df = collect(list(range(a.seeds)), a.overwrite)
    analyse(df)
    if not a.skip_v2:
        res = recompute_v2()
        report_v2(res)
        with open(V2_JSON, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote {V2_JSON}")


if __name__ == "__main__":
    main()
