"""
SHAP analysis of the trained Isolation Forest.

Usage (from the project root):
    python3 scripts/shap_analysis.py --artifacts model.pkl --normal normal.pcap --attack capture.pcap --out shap_out

Writes: shap_summary.png, shap_block_top_features.png, shap_waterfall_top_block.png, shap_block_top_features.csv
"""
from __future__ import annotations

import argparse
import io
import contextlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nad.features import featurize_for_ml  # noqa: E402
from nad.pipeline import build_labeled_frame, load_artifacts, score_pcap  # noqa: E402


def features(pcap: str, art) -> pd.DataFrame:
    df = build_labeled_frame(pcap, art.cfg)
    X, _ = featurize_for_ml(df)
    return X.reindex(columns=art.feature_names, fill_value=0).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="model.pkl")
    ap.add_argument("--normal", default="normal.pcap")
    ap.add_argument("--attack", default="capture.pcap")
    ap.add_argument("--out", default="shap_out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    art = load_artifacts(a.artifacts)
    pipe = art.detector.pipeline_if
    scaler, iforest = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    names = art.feature_names

    Xn, Xa = features(a.normal, art), features(a.attack, art)
    with contextlib.redirect_stdout(io.StringIO()):
        dec = score_pcap(a.attack, art, explain=False)

    expl = shap.TreeExplainer(iforest)
    base = float(np.ravel(expl.expected_value)[0])
    svn = np.asarray(expl.shap_values(scaler.transform(Xn)))
    sva = np.asarray(expl.shap_values(scaler.transform(Xa)))

    # Sanity check: base + sum(SHAP) == E[path length] reconstructed from sklearn's score_samples
    n = iforest.max_samples_
    c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    eh = -c * np.log2(-iforest.score_samples(scaler.transform(Xa)))
    print(f"additivity max error: {np.abs(sva.sum(1) + base - eh).max():.2e}")
    thr = -c * np.log2(-iforest.offset_)

    # 1) Summary (beeswarm): all normal flows + an equal-size random sample of attack-capture flows
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xa), min(len(Xn), len(Xa)), replace=False)
    plt.figure()
    shap.summary_plot(np.vstack([svn, sva[idx]]), pd.concat([Xn, Xa.iloc[idx]]), max_display=12, show=False)
    plt.xlabel("SHAP value (change in average isolation path length)\n<- more anomalous | more normal ->")
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close("all")

    # 2) Mean SHAP on BLOCK decisions
    blk = (dec["action"] == "BLOCK").values
    push = -sva[blk].mean(0)
    tbl = pd.DataFrame({"feature": names, "mean_path_reduction": push})
    tbl["share_pct"] = np.where(push > 0, 100 * push / push[push > 0].sum(), 0)
    tbl = tbl.sort_values("mean_path_reduction", ascending=False)
    tbl.to_csv(out / "shap_block_top_features.csv", index=False)
    print(f"BLOCK flows: {blk.sum()} | path length {base:.2f} -> {(sva[blk].sum(1) + base).mean():.2f} (threshold {thr:.2f})")
    print(tbl.head(5).round(3).to_string(index=False))
    top = tbl.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(top["feature"], top["mean_path_reduction"], color="#2a78d6", height=0.6)
    for y, (v, s) in enumerate(zip(top["mean_path_reduction"], top["share_pct"])):
        ax.text(v + 0.04, y, f"{v:.2f} ({s:.0f}%)", va="center", fontsize=9.5)
    ax.set_xlabel("Mean reduction in isolation path length (splits)")
    ax.set_title(f"What drove the {blk.sum():,} BLOCK decisions", loc="left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "shap_block_top_features.png", dpi=180)
    plt.close("all")

    # 3) Waterfall for the highest-risk BLOCK flow from the most frequent blocked source
    if blk.any():
        src = dec.loc[blk, "src_ip"].value_counts().index[0]
        i = dec[(dec["src_ip"] == src) & blk].index[0]
        e = shap.Explanation(values=sva[i], base_values=base, data=Xa.loc[i].values, feature_names=names)
        plt.figure()
        shap.plots.waterfall(e, max_display=10, show=False)
        plt.title(f"Flow {i}: {src} -> port {dec.loc[i, 'dst_port']} (risk {dec.loc[i, 'risk_score']:.0f})", loc="left")
        plt.savefig(out / "shap_waterfall_top_block.png", dpi=180, bbox_inches="tight")
        plt.close("all")
    print("saved to", out)


if __name__ == "__main__":
    main()
