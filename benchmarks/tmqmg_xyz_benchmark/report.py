# ruff: noqa: I001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tmQMg benchmark accuracy and timing.")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more benchmark results.csv paths.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching --results order. Defaults to file stem.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for plots.")
    return parser.parse_args()


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "equivalent" not in df.columns or "timing_ms_total" not in df.columns:
        raise ValueError(f"missing required columns in {path}")
    return df


def _normalize_results(
    paths: list[Path], labels: list[str] | None
) -> list[tuple[str, pd.DataFrame]]:
    if labels is not None and len(labels) != len(paths):
        raise ValueError("--labels must match --results length")
    normalized: list[tuple[str, pd.DataFrame]] = []
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else path.parent.name
        normalized.append((label, _load_results(path)))
    return normalized


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).map(
        lambda value: (
            value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}
        )
    )


def _comparable_mask(df: pd.DataFrame) -> pd.Series:
    if "reference_status" in df.columns:
        return cast(pd.Series, df["reference_status"]).eq("valid")
    status_mask = df["status"].fillna("") != "skipped" if "status" in df.columns else True
    if "comparison_skipped" not in df.columns:
        return status_mask & df["equivalent"].notna()
    return status_mask & ~_truthy_series(cast(pd.Series, df["comparison_skipped"]))


def _equivalent_mask(df: pd.DataFrame) -> pd.Series:
    if "evaluator_decision" in df.columns:
        return df["status"].eq("ok") & df["evaluator_decision"].eq("equivalent")
    return _truthy_series(cast(pd.Series, df["equivalent"]))


def _accuracy_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_id, group in df.groupby("method_id", sort=True):
        denominator = group.loc[_comparable_mask(group)]
        valid = _equivalent_mask(denominator)
        accuracy = float(valid.mean()) if len(denominator) else 0.0
        rows.append({"method_id": method_id, "accuracy": accuracy, "count": len(denominator)})
    return pd.DataFrame(rows)


def _plot_accuracy(datasets: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt = _pyplot()
    method_ids = sorted({method_id for _, df in datasets for method_id in df["method_id"].unique()})
    summary = {label: _accuracy_by_method(df).set_index("method_id") for label, df in datasets}

    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / max(len(datasets), 1)
    offsets = [(-0.4 + width / 2.0) + i * width for i in range(len(datasets))]
    colors = ["#2F6FDF", "#D95F02", "#1B9E77", "#7570B3"]
    for index, (label, _) in enumerate(datasets):
        values = [
            float(summary[label].loc[m, "accuracy"]) if m in summary[label].index else 0.0
            for m in method_ids
        ]
        counts = [
            int(summary[label].loc[m, "count"]) if m in summary[label].index else 0
            for m in method_ids
        ]
        positions = [i + offsets[index] for i in range(len(method_ids))]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=label,
            color=colors[index % len(colors)],
        )
        for bar, count in zip(bars, counts, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.98, bar.get_height() + 0.02),
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(range(len(method_ids)))
    ax.set_xticklabels(method_ids, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Method")
    ax.set_title("tmQMg Accuracy by Method")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="OpenBabel version")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_timing_boxplot(datasets: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt = _pyplot()
    method_ids = sorted({method_id for _, df in datasets for method_id in df["method_id"].unique()})
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#2F6FDF", "#D95F02", "#1B9E77", "#7570B3"]
    width = 0.35 if len(datasets) > 1 else 0.5
    for index, (label, df) in enumerate(datasets):
        plot_df = df.loc[
            (df["status"] != "skipped") & df["timing_ms_total"].notna(),
            ["method_id", "timing_ms_total"],
        ]
        offset = (index - (len(datasets) - 1) / 2.0) * width
        positions = [method_index + offset for method_index in range(len(method_ids))]
        series = [
            plot_df.loc[plot_df["method_id"] == method_id, "timing_ms_total"].tolist()
            for method_id in method_ids
        ]
        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=width * 0.8,
            showfliers=False,
            patch_artist=True,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(colors[index % len(colors)])
            patch.set_alpha(0.35)
        for median in boxplot["medians"]:
            median.set_color(colors[index % len(colors)])
        for element in ("whiskers", "caps"):
            for line in boxplot[element]:
                line.set_color(colors[index % len(colors)])
        boxplot["boxes"][0].set_label(label)

    ax.set_xticks(range(len(method_ids)))
    ax.set_xticklabels(method_ids, rotation=20)
    ax.set_ylabel("Timing (ms)")
    ax.set_xlabel("Method")
    ax.set_title("tmQMg Timing Distribution by Method")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="OpenBabel version")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_timing_recall_boxplot(datasets: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    plt = _pyplot()
    method_ids = sorted({method_id for _, df in datasets for method_id in df["method_id"].unique()})
    dataset_labels = [label for label, _ in datasets]
    grouped = len(datasets) > 1
    fig, ax = plt.subplots(figsize=(9.0, 12.0))
    colors = ["#2F6FDF", "#D95F02", "#1B9E77", "#7570B3"]
    width = 0.35 if grouped else 0.55
    all_positive_timings: list[float] = []

    for index, (label, df) in enumerate(datasets):
        offset = (index - (len(datasets) - 1) / 2.0) * width
        positions = [method_index + offset for method_index in range(len(method_ids))]
        series = []
        recalls: dict[str, float] = {}
        counts: dict[str, int] = {}
        for method_id in method_ids:
            method_df = df.loc[df["method_id"] == method_id]
            timing = method_df.loc[
                (method_df["status"] != "skipped") & method_df["timing_ms_total"].notna(),
                "timing_ms_total",
            ].astype(float)
            timing = timing.loc[timing > 0.0]
            values = timing.tolist()
            series.append(values)
            all_positive_timings.extend(values)
            denominator = method_df.loc[_comparable_mask(method_df)]
            recalls[method_id] = (
                float(_equivalent_mask(denominator).mean()) if len(denominator) else 0.0
            )
            counts[method_id] = len(denominator)

        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=width * 0.82,
            vert=False,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(colors[index % len(colors)])
            patch.set_alpha(0.38)
        for median in boxplot["medians"]:
            median.set_color("#111111")
            median.set_linewidth(1.4)
        for element in ("whiskers", "caps"):
            for line in boxplot[element]:
                line.set_color(colors[index % len(colors)])
        if boxplot["boxes"]:
            boxplot["boxes"][0].set_label(label)

        for method_id, position in zip(method_ids, positions, strict=False):
            ax.text(
                1.015,
                position,
                f"{recalls[method_id] * 100:.1f}% ({int(recalls[method_id] * counts[method_id])}/{counts[method_id]})",
                ha="left",
                va="center",
                fontsize=8,
                transform=ax.get_yaxis_transform(),
            )

    if not all_positive_timings:
        raise ValueError("no positive timing_ms_total values to plot")

    ax.set_xscale("log")
    ymin = min(all_positive_timings)
    ymax = max(all_positive_timings)
    ax.set_xlim(max(ymin * 0.55, 1e-3), ymax * 3.4)
    ax.set_yticks(range(len(method_ids)))
    ax.set_yticklabels(method_ids)
    ax.set_xlabel("Processing time (ms, log scale)")
    ax.set_ylabel("Method")
    ax.set_title("tmQMg First 1000: Timing Distribution and Recall")
    ax.grid(axis="x", which="both", alpha=0.22)
    if grouped:
        ax.legend(title="Dataset")
    elif dataset_labels:
        ax.text(
            0.995,
            0.02,
            dataset_labels[0],
            ha="right",
            va="bottom",
            fontsize=9,
            transform=ax.transAxes,
        )
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.22, right=0.82)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    datasets = _normalize_results(args.results, args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot_accuracy(datasets, args.out_dir / "tmqmg_accuracy.png")
    _plot_timing_boxplot(datasets, args.out_dir / "tmqmg_timing_boxplot.png")
    _plot_timing_recall_boxplot(datasets, args.out_dir / "tmqmg_timing_recall_boxplot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
