#!/usr/bin/env python3
# analyze_fg_error.py
# -------------------------------------------------------------
# Analyze relationship between prediction error (difference)
# and functional groups in parity CSV produced by pka_prediction_refactored.py
# -------------------------------------------------------------
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns   # ← 若想用 seaborn 箱形圖, 取消註釋

# -------------------------------------------------------------
def _explode_groups(df: pd.DataFrame) -> pd.DataFrame:
    """把 'A;B' 這類多重官能基拆成多列"""
    return (
        df.assign(functional_group=df["functional_group"].str.split(";"))
          .explode("functional_group")
          .dropna(subset=["functional_group"])
    )


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """回傳 per-group 誤差統計 DataFrame"""
    return (
        df.groupby("functional_group")["difference"]
          .agg(count="count",
               mean="mean",
               median="median",
               std="std",
               min="min",
               max="max")
          .reset_index()
          .sort_values("mean")
    )


def plot_box(df: pd.DataFrame, groups: list[str], out_png: pathlib.Path) -> None:
    """畫差值箱形圖（matplotlib 原生；無 seaborn 依賴）"""
    data = [df.loc[df["functional_group"] == g, "difference"].values for g in groups]

    plt.figure(figsize=(10, max(4, 0.4 * len(groups))))
    plt.boxplot(data, vert=False, labels=groups, showfliers=False)
    plt.axvline(0, color="k", lw=1, alpha=0.7)
    plt.xlabel("Absolute error (difference)")
    plt.ylabel("Functional group")
    plt.title("Prediction error by functional group")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Analyze difference vs. functional_group in parity CSV"
    )
    ap.add_argument("-v","--version", type=str, default='pka_ver10',
                    help="model version")
    ap.add_argument("-o", "--out-dir", type=pathlib.Path,
                    default=pathlib.Path("./fg_analysis"),
                    help="directory to save summary/fig (default: ./fg_analysis)")
    ap.add_argument("--min-count", type=int, default=5,
                    help="only keep groups with >= MIN_COUNT samples for boxplot (default: 5)")
    args = ap.parse_args(argv)
    args.csv = f'../results/{args.version}/pka_combined_prediction_{args.version}.csv'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 讀檔 + 清理
    df_raw = pd.read_csv(args.csv)
    df = df_raw.dropna(subset=["difference", "functional_group"])
    df = _explode_groups(df)

    if df.empty:
        sys.exit("No rows with both difference and functional_group")

    # 2) 統計表
    summary = make_summary(df)
    summary_path = args.out_dir / "fg_error_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] summary → {summary_path.resolve()}")

    # 3) 箱形圖
    valid_groups = summary.loc[summary["count"] >= args.min_count, "functional_group"].tolist()
    if valid_groups:
        boxplot_path = args.out_dir / "fg_error_boxplot.png"
        plot_box(df, valid_groups, boxplot_path)
        print(f"[INFO] boxplot → {boxplot_path.resolve()}")
    else:
        print(f"[WARN] no group meets min-count = {args.min_count}; skip boxplot")


if __name__ == "__main__":
    main() 