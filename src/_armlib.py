"""
Shared arm construction, training and recording for R-5 / X-3 / B-1.

One implementation, tested once, rather than the same twenty lines copied into
three scripts.  All of it goes through _r1core, so the channel decomposition is
verified against main.py's own episode return on every iteration of every run.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

from _r1core import channel_aucs, channels, instrument_trainer

BASELINE = "baseline_ppo"
ALGEBRA_OF = {"so": "so", "sl": "sl", "gl": "full", "sym": "sym"}


# --------------------------------------------------------------------------- #
class Sink:
    """Append-as-you-go CSV writer; a crash never costs completed runs."""

    def __init__(self, path, cols):
        # Truncates unconditionally.  guard_outputs() is what protects a
        # previous sweep, and every caller runs it first, before the model
        # load.  There is no resume path: after a crash you re-run with
        # --overwrite, which is why the guard exists.
        self.path, self.cols, self.n = path, cols, 0
        pd.DataFrame(columns=cols).to_csv(path, index=False)

    def add(self, rows):
        if rows:
            pd.DataFrame(rows, columns=self.cols).to_csv(
                self.path, mode="a", header=False, index=False)
            self.n += len(rows)


def _main_hash() -> str:
    import hashlib
    try:
        with open("main.py", "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:12]
    except OSError:
        return "unavailable"


def write_meta(csv_path: str, meta: dict) -> None:
    """Record the settings a later --analyse-only cannot otherwise know."""
    import json
    meta = dict(meta)
    meta["main_py_md5"] = _main_hash()
    with open(csv_path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def check_meta(csv_path: str, expected: dict) -> None:
    """Refuse to analyse a CSV produced under different settings.

    Without this, --analyse-only on a stale file would report numbers from
    a different grid, a different geometry weight, or a different main.py.
    """
    import json
    p = csv_path + ".meta.json"
    if not os.path.exists(p):
        print(f"!! WARNING: no {p}; cannot verify that {csv_path} was produced "
              f"under the settings this analysis assumes.")
        return
    with open(p) as fh:
        got = json.load(fh)
    bad = {k: (got.get(k), v) for k, v in expected.items()
           if str(got.get(k)) != str(v)}
    if bad:
        lines = "\n".join(f"    {k}: file={a!r}  now={b!r}" for k, (a, b) in bad.items())
        sys.exit(f"{csv_path} was produced under different settings:\n{lines}\n"
                 f"Re-run the sweep, or analyse it with the matching version.")
    cur = _main_hash()
    if got.get("main_py_md5") not in (cur, "unavailable") and cur != "unavailable":
        print(f"!! WARNING: main.py has changed since {csv_path} was written "
              f"({got.get('main_py_md5')} -> {cur}).")


def guard_outputs(paths, overwrite: bool) -> None:
    """Refuse to truncate a previous sweep -- checked before the model load."""
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0 and not overwrite:
            sys.exit(f"{p} exists and is non-empty.  Refusing to truncate a "
                     f"previous sweep.\nMove it aside, or re-run with --overwrite.")


# --------------------------------------------------------------------------- #
def build(env, arm: str, seed: int, state_dim: int, n_actions: int, k: int,
          theta_lr=None, geo_aux_coef=None):
    """Construct (policy, trainer) for `arm`, seeded immediately beforehand.

    `arm` is BASELINE or a key of ALGEBRA_OF.  Re-seeding happens here, right
    before construction, so no arm inherits another's random-number state --
    the artefact that makes main.py's own seed-0 so(32) score differ between
    its two runners.
    """
    from main import (LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      _default_cfg, _seed_all)

    if theta_lr is not None and arm == BASELINE:
        raise ValueError("theta_lr is meaningless for the baseline arm: it has "
                         "no theta, and LieStructuredPPO drops cfg.theta_lr "
                         "entirely on that branch.  Passing it would add a "
                         "grid dimension that has no effect.")
    _seed_all(seed)
    cfg = _default_cfg()
    if theta_lr is not None:
        cfg.theta_lr = float(theta_lr)
    if geo_aux_coef is not None:
        cfg.geo_aux_coef = float(geo_aux_coef)

    if arm == BASELINE:
        policy = BaselinePolicy(state_dim, n_actions)
        trainer = LieStructuredPPO(env, policy, ValueNet(state_dim), cfg,
                                   use_lie_projection=False)
    else:
        if arm not in ALGEBRA_OF:
            raise ValueError(f"unknown arm {arm!r}; expected {BASELINE} or "
                             f"one of {sorted(ALGEBRA_OF)}")
        alg = ALGEBRA_OF[arm]
        policy = LiePolicy(state_dim, n_actions, k=k, algebra=alg)
        trainer = LieStructuredPPO(env, policy, ValueNet(state_dim), cfg,
                                   use_lie_projection=True, algebra=alg)
    return policy, trainer, cfg


def run_one(env, arm, seed, state_dim, n_actions, k, w, horizon,
            theta_lr=None, geo_aux_coef=None, extra=None):
    """Train one arm and return a record dict.  Never raises for a training
    failure: the failure is recorded so the sweep continues.  KeyboardInterrupt
    is not swallowed."""
    base = dict(arm=arm, seed=seed, status="ok")
    if extra:
        base.update(extra)
    try:
        policy, trainer, _ = build(env, arm, seed, state_dim, n_actions, k,
                                   theta_lr, geo_aux_coef)
        instrument_trainer(trainer, env, None, horizon, strict=True)
        res = trainer.train(verbose=False)
    except KeyboardInterrupt:
        raise
    except Exception as e:                                   # noqa: BLE001
        base.update(status=f"{type(e).__name__}: {e}"[:120])
        return base, None, None

    pairs = channels(res, trainer)
    rt = float(np.mean([p[0] for p in pairs]))
    rg = float(np.mean([p[1] for p in pairs]))
    _, _, a_tot = channel_aucs(pairs, w, horizon)
    auc = float(res["auc"])
    recon = abs(a_tot - auc)
    spec = [s for s in res.get("spectral_radii", [])
            if s is not None and math.isfinite(s)]
    finite = all(math.isfinite(x) for x in (rt, rg, auc, recon))
    base.update(r_task=rt, r_geo=rg, auc=auc,
                rho=float(np.median(spec)) if spec else np.nan,
                recon_err=recon,
                status="ok" if finite else "non-finite")
    return base, pairs, policy


# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, col: str, by: str, arms) -> pd.DataFrame:
    """Drop non-finite rows and any block missing an arm, keeping the design
    paired.  `by` is the blocking variable (seed, or env draw)."""
    d = df[df.status == "ok"].copy()
    dup = d.duplicated([by, "arm"]).sum()
    if dup:
        print(f"  !! {dup} duplicated ({by}, arm) row(s); keeping the last of "
              f"each.  Two partial sweeps were probably concatenated.")
        d = d.drop_duplicates([by, "arm"], keep="last")
    bad = ~np.isfinite(pd.to_numeric(d[col], errors="coerce"))
    if bad.any():
        print(f"  dropping {int(bad.sum())} non-finite '{col}' row(s)")
        d = d.loc[~bad]
    counts = d.groupby(by)["arm"].nunique()
    full = counts[counts == len(arms)].index
    lost = sorted(set(d[by]) - set(full))
    if lost:
        print(f"  dropping {len(lost)} incomplete block(s) to keep the design "
              f"paired: {lost}")
    return d[d[by].isin(full)]


def paired(df, a, b, col, by):
    """a - b, matched on the blocking variable."""
    x = df[df.arm == a].set_index(by)[col].astype(float)
    y = df[df.arm == b].set_index(by)[col].astype(float)
    idx = x.index.intersection(y.index)
    d = (x.loc[idx] - y.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return d.to_numpy(), len(d)


def report(rows, title, note=""):
    """Holm-correct a family of paired comparisons and print it."""
    from sppg_defense.stats.tests import holm, paired_report

    out, pvals, notes = [], [], []
    for r in rows:
        d, n = np.asarray(r["d"], float), len(r["d"])
        meta = {k: v for k, v in r.items() if k != "d"}
        if n < 3:
            out.append(dict(**meta, n=n, degenerate="n<3"))
            pvals.append(1.0)
            continue
        if float(np.std(d, ddof=1)) == 0.0:
            # Zero-variance differences make paired_report return p=0, dz=inf
            # and power=1 -- a maximally confident result from data with no
            # information, so it is marked untestable instead.
            out.append(dict(**meta, n=n, diff=float(d.mean()),
                            degenerate="zero variance"))
            pvals.append(1.0)
            notes.append(f"{meta.get('label','?')}: all {n} paired differences "
                         f"are identical ({d.mean():+.4g}); no test is possible")
            continue
        pr = paired_report(d, np.zeros_like(d))
        if not np.isfinite(pr.t_p):
            out.append(dict(**meta, n=n, diff=pr.diff, degenerate="non-finite p"))
            pvals.append(1.0)
            notes.append(f"{meta.get('label','?')}: the paired t returned a "
                         f"non-finite p-value; reported as untestable, not as "
                         f"'not significant'")
            continue
        out.append(dict(**meta, n=n, diff=pr.diff, ci=pr.diff_ci, dz=pr.cohen_dz,
                        p=pr.t_p, power=pr.achieved_power))
        pvals.append(pr.t_p)
    hp = holm(pvals, 0.05)
    for o, ph, rj in zip(out, hp["p_holm"], hp["reject"]):
        o["p_holm"], o["sig"] = ph, rj

    print("\n" + "=" * 96)
    print(f"{title}   Holm-corrected over {len(out)} comparisons")
    if note:
        print(note)
    print("=" * 96)
    for o in out:
        lab = o.get("label", "")
        if o.get("degenerate"):
            v = f"  diff={o['diff']:+.4f}" if "diff" in o else ""
            print(f"  {lab:<44s} n={o['n']}  UNTESTABLE ({o['degenerate']}){v}")
            continue
        if "diff" not in o:
            print(f"  {lab:<44s} n={o['n']}  (insufficient blocks)")
            continue
        print(f"  {lab:<44s} n={o['n']:>2d}  {o['diff']:>+9.4f}"
              f"{'*' if o['sig'] else ' '} "
              f"[{o['ci'][0]:>+8.4f},{o['ci'][1]:>+8.4f}]  dz={o['dz']:>+6.2f}  "
              f"p_holm={o['p_holm']:>9.2e}  "
              f"power={(o['power'] if o['power'] is not None else float('nan')):.2f}")
    for nt in notes:
        print(f"  !! {nt}")
    print("  * = significant after Holm.  'power' is post-hoc power at the "
          "observed effect --\n  a restatement of p that says nothing about a null.")
    return out


def equivalent(o, margin: float) -> bool:
    """True only if the whole CI lies inside +/- margin.

    'Not significant' does not establish equivalence.  Any verdict branch that
    treats two arms as the same goes through this.
    """
    if not o or "ci" not in o:
        return False
    lo, hi = o["ci"]
    return bool(abs(lo) < margin and abs(hi) < margin)
