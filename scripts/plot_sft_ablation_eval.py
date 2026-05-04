#!/usr/bin/env python3
"""
Plot SFT ablation evals under ``per_size_<N>/eval_after_sft.json``.

Default: **x = SFT ``per_size``**, **y = solve rate**; one curve per board side length,
coloured by board size (continuous **colorbar**, paper-friendly). **Mean** over boards as a
bold dashed line (legend entry only for the mean).

``--x-axis board``: x = board size, one curve per ``per_size`` (colorbar = SFT budget).

**Scaling (``--x-axis per_size`` only):** fits ``log10(y_clip) ≈ β·log10(N) + α`` (``y`` = solve rate),
i.e. ``y ∝ N^β`` in the large-data regime up to clipping; writes ``sft_scaling_stats.json`` under
``--root`` unless ``--no-scaling-json``. Use ``--show-mean-powerlaw`` to overlay the mean-curve fit.

Example::

    python scripts/plot_sft_ablation_eval.py --root runs/sft_ablation \\
        --out runs/sft_ablation/solve_rate_vs_sft_per_size.png

Saving ``.png`` also writes a vector ``.pdf`` next to it (disable with ``--no-auto-pdf``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Normalize


def _parse_budget(dir_name: str) -> int | None:
    m = re.fullmatch(r"per_size_(\d+)", dir_name)
    return int(m.group(1)) if m else None


def load_runs(root: Path) -> list[tuple[int, dict[int, float], float]]:
    """Sorted by budget: (per_size, {board_side -> solve_rate}, mean over all boards)."""
    root = root.resolve()
    runs: list[tuple[int, dict[int, float], float]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        budget = _parse_budget(child.name)
        if budget is None:
            continue
        js = child / "eval_after_sft.json"
        if not js.is_file():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        m: dict[int, float] = {}
        for item in data.get("per_size", []):
            m[int(item["size"])] = float(item["solve_rate"])
        if not m:
            continue
        if "mean_solve_rate_across_sizes" in data:
            mean_all = float(data["mean_solve_rate_across_sizes"])
        else:
            mean_all = sum(m.values()) / len(m)
        runs.append((budget, m, mean_all))
    runs.sort(key=lambda t: t[0])
    return runs


def _loglog_linear_fit(
    n_values: list[float],
    y_values: list[float],
    *,
    y_floor: float = 1e-6,
) -> dict[str, Any] | None:
    """Fit ``log10(max(y,y_floor)) = beta * log10(N) + alpha``; return coeffs and R² in log-space."""

    if len(n_values) < 2 or len(n_values) != len(y_values):
        return None
    n_arr = np.asarray(n_values, dtype=np.float64)
    y_arr = np.maximum(np.asarray(y_values, dtype=np.float64), float(y_floor))
    if np.any(n_arr <= 0):
        return None
    log_n = np.log10(n_arr)
    log_y = np.log10(y_arr)
    if np.std(log_n) < 1e-15 or np.std(log_y) < 1e-15:
        return None
    beta, alpha = np.polyfit(log_n, log_y, 1)
    log_y_hat = beta * log_n + alpha
    ss_res = float(np.sum((log_y - log_y_hat) ** 2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-20 else float("nan")
    return {
        "beta": float(beta),
        "alpha": float(alpha),
        "r2_log10_space": float(r2),
        "interpretation": "solve_rate ≈ 10**alpha * N**beta (in log-log linear regime; y floored for zeros)",
        "y_floor": float(y_floor),
    }


def build_scaling_stats(runs: list[tuple[int, dict[int, float], float]]) -> dict[str, Any]:
    """Tables + log-log fits for mean and per-board-size solve rates vs SFT budget ``N``."""

    budgets = [float(b) for b, _, _ in runs]
    means = [float(mn) for _, _, mn in runs]
    raw_rows = [{"N": int(b), "mean_solve_rate_across_sizes": mn} for b, _, mn in runs]

    out: dict[str, Any] = {
        "n_runs": len(runs),
        "budgets": [int(b) for b, _, _ in runs],
        "mean_solve_rates": means,
        "fit_model": "log10(max(solve_rate, y_floor)) = beta * log10(N) + alpha  <=>  solve_rate ≈ 10**alpha * N**beta",
        "mean_across_sizes_fit": _loglog_linear_fit(budgets, means),
        "per_board_size": {},
        "raw_rows": raw_rows,
    }

    all_sizes = sorted({sz for _, m, _ in runs for sz in m})
    for sz in all_sizes:
        ys = [float(m[sz]) for _, m, _ in runs if sz in m]
        ns = [float(b) for b, m, _ in runs if sz in m]
        if len(ns) >= 2:
            out["per_board_size"][str(sz)] = {
                "solve_rates": ys,
                "fit": _loglog_linear_fit(ns, ys),
            }

    # Simple deltas: mean gain when doubling budget between consecutive measured N (not log-uniform).
    gains: list[dict[str, Any]] = []
    for i in range(1, len(runs)):
        n0, y0 = runs[i - 1][0], runs[i - 1][2]
        n1, y1 = runs[i][0], runs[i][2]
        if n0 > 0 and n1 > n0:
            gains.append(
                {
                    "N_low": int(n0),
                    "N_high": int(n1),
                    "ratio_N": float(n1 / n0),
                    "mean_solve_low": float(y0),
                    "mean_solve_high": float(y1),
                    "delta_mean": float(y1 - y0),
                }
            )
    out["consecutive_budget_mean_deltas"] = gains

    return out


def _mean_powerlaw_curve(
    runs: list[tuple[int, dict[int, float], float]],
    *,
    y_floor: float = 1e-6,
) -> tuple[list[float], list[float], dict[str, Any]] | None:
    """Return (budgets, y_hat on budgets) for mean solve rate if fit exists."""

    budgets = [float(b) for b, _, _ in runs]
    means = [float(mn) for _, _, mn in runs]
    fit = _loglog_linear_fit(budgets, means, y_floor=y_floor)
    if fit is None:
        return None
    beta = fit["beta"]
    alpha = fit["alpha"]
    log_n = np.log10(np.asarray(budgets, dtype=np.float64))
    y_hat = np.power(10.0, beta * log_n + alpha)
    y_hat = np.clip(y_hat, 0.0, 1.0)
    return budgets, y_hat.tolist(), fit


def _print_scaling_summary(stats: dict[str, Any]) -> None:
    mfit = stats.get("mean_across_sizes_fit")
    print("[scaling] mean solve rate vs SFT budget N:", flush=True)
    if mfit:
        print(
            f"  log10 fit: beta={mfit['beta']:.4f}, alpha={mfit['alpha']:.4f}, "
            f"R²(log)={mfit['r2_log10_space']:.4f}  =>  rate ≈ 10^{mfit['alpha']:.3f} * N^{mfit['beta']:.3f}",
            flush=True,
        )
    else:
        print("  (not enough variation or points for mean log-log fit)", flush=True)
    per = stats.get("per_board_size") or {}
    good = [(sz, d["fit"]) for sz, d in per.items() if d.get("fit")]
    if good:
        print("[scaling] per-board log-log β (solve_rate ~ N^β), R²(log):", flush=True)
        for sz, f in sorted(good, key=lambda t: int(t[0])):
            print(
                f"  board {sz}: beta={f['beta']:.4f}, R²={f['r2_log10_space']:.4f}",
                flush=True,
            )


def _apply_paper_rc() -> None:
    """Fonts and lines suitable for print / two-column papers."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.38,
            "grid.linestyle": "-",
        }
    )


def _get_cmap(name: str):
    cm = getattr(mpl, "colormaps", None)
    if cm is not None:
        try:
            return cm[name]
        except KeyError:
            pass
    return mpl.cm.get_cmap(name)


def _plot_per_size_mode(
    ax: plt.Axes,
    runs: list[tuple[int, dict[int, float], float]],
    *,
    log_x: bool,
    title: str | None,
    mean_powerlaw: tuple[list[float], list[float], dict[str, Any]] | None = None,
) -> None:
    budgets = [b for b, _, _ in runs]
    all_sizes = sorted({sz for _, m, _ in runs for sz in m})
    sz_min, sz_max = float(all_sizes[0]), float(all_sizes[-1])
    norm = Normalize(vmin=sz_min, vmax=sz_max)
    # Perceptually uniform; prints legibly in greyscale.
    cmap = _get_cmap("cividis")

    for sz in all_sizes:
        xs = [b for b, m, _ in runs if sz in m]
        ys = [m[sz] for b, m, _ in runs if sz in m]
        c = cmap(norm(float(sz)))
        ax.plot(xs, ys, marker="o", ms=3.0, lw=1.25, color=c, alpha=0.92)

    mean_ys = [mean_all for _, _, mean_all in runs]
    (mean_line,) = ax.plot(
        budgets,
        mean_ys,
        linestyle=(0, (4, 2.2)),
        color="#111111",
        lw=2.15,
        marker="D",
        ms=3.8,
        mfc="white",
        mec="#111111",
        mew=1.0,
        zorder=10,
        label="Mean (all sides)",
    )

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel(r"SFT budget $N$ (max.\ puzzles per training side length)")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", alpha=0.2)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.82, pad=0.02, aspect=22)
    cbar.set_label("Board side length", rotation=270, labelpad=14)
    cbar.ax.tick_params(length=2.5, width=0.6)

    handles: list[Any] = [mean_line]
    labels: list[str] = ["Mean (all sides)"]
    if mean_powerlaw is not None:
        _bs, _yh, fit = mean_powerlaw
        r2 = float(fit.get("r2_log10_space", float("nan")))
        (fit_line,) = ax.plot(
            _bs,
            _yh,
            linestyle=(0, (3, 2)),
            color="#b35900",
            lw=1.85,
            alpha=0.9,
            zorder=9,
            clip_on=False,
        )
        handles.append(fit_line)
        beta = float(fit["beta"])
        alpha = float(fit["alpha"])
        # Mathtext line 2: avoid f-string parsing `{` inside `\mathrm{log}`.
        coef_line = (
            "$"
            + rf"\beta={beta:.3f},\ \alpha={alpha:.3f},\ "
            + r"R^2_{\mathrm{log}}="
            + f"{r2:.3f}$"
        )
        labels.append(
            r"$\log_{10}\max(r,10^{-6})=\beta\log_{10}N+\alpha$" + "\n" + coef_line
        )

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="0.55",
        facecolor="white",
        framealpha=0.94,
    )
    leg.get_frame().set_linewidth(0.6)

    if title:
        ax.set_title(title, pad=8)


def _plot_board_mode(
    ax: plt.Axes,
    runs: list[tuple[int, dict[int, float], float]],
    *,
    title: str | None,
) -> None:
    budgets = [b for b, _, _ in runs]
    b_min, b_max = float(budgets[0]), float(budgets[-1])
    norm = Normalize(vmin=b_min, vmax=b_max)
    cmap = _get_cmap("plasma")

    for budget, m, _ in runs:
        xs = sorted(m)
        ys = [m[s] for s in xs]
        c = cmap(norm(float(budget)))
        ax.plot(xs, ys, marker="o", ms=3.0, lw=1.25, color=c, alpha=0.92, clip_on=False)

    ax.set_xlabel("Board side length")
    ax.set_ylabel("Solve rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, which="major", axis="both")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.82, pad=0.02, aspect=22)
    cbar.set_label(r"SFT budget $N$", rotation=270, labelpad=12)
    cbar.ax.tick_params(length=2.5, width=0.6)

    if title:
        ax.set_title(title, pad=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("runs/sft_ablation"), help="Directory containing per_size_* folders.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (.png or .pdf). Default: <root>/solve_rate_vs_sft_per_size.png",
    )
    p.add_argument(
        "--x-axis",
        choices=["per_size", "board"],
        default="per_size",
        help="per_size: x=SFT budget, colour=board size. board: x=board size, colour=budget.",
    )
    p.add_argument(
        "--linear-x",
        action="store_true",
        help="Force linear x for per_size mode (default is log when max/min budget ≥ 4).",
    )
    p.add_argument(
        "--log-x",
        action="store_true",
        help="Force log x for per_size mode (overrides --linear-x).",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title (omit for no title).")
    p.add_argument("--dpi", type=int, default=220, help="Raster resolution if saving PNG.")
    p.add_argument(
        "--no-auto-pdf",
        action="store_true",
        help="When saving .png, do not also write a sibling .pdf.",
    )
    p.add_argument(
        "--no-scaling-json",
        action="store_true",
        help="Do not write sft_scaling_stats.json (scaling summary still prints unless --quiet-scaling).",
    )
    p.add_argument(
        "--scaling-json",
        type=Path,
        default=None,
        help="Path for scaling stats JSON (default: <root>/sft_scaling_stats.json).",
    )
    p.add_argument(
        "--quiet-scaling",
        action="store_true",
        help="Suppress printed scaling summary (still writes JSON unless --no-scaling-json).",
    )
    p.add_argument(
        "--show-mean-powerlaw",
        action="store_true",
        help="With --x-axis per_size, overlay log–log least-squares fit on the mean solve-rate curve.",
    )
    args = p.parse_args()

    root = args.root
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    out = args.out
    if out is None:
        out = (
            root / ("solve_rate_vs_sft_per_size.png" if args.x_axis == "per_size" else "solve_rate_vs_board_size.png")
        )
    elif not out.is_absolute():
        out = (Path.cwd() / out).resolve()

    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no per_size_*/eval_after_sft.json found under {root}")

    stats = build_scaling_stats(runs)
    scaling_path = args.scaling_json if args.scaling_json is not None else (root / "sft_scaling_stats.json")
    if not scaling_path.is_absolute():
        scaling_path = (Path.cwd() / scaling_path).resolve()
    if not args.no_scaling_json:
        scaling_path.parent.mkdir(parents=True, exist_ok=True)
        scaling_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"wrote scaling stats: {scaling_path}", flush=True)
    if not args.quiet_scaling:
        _print_scaling_summary(stats)

    mean_pl: tuple[list[float], list[float], dict[str, Any]] | None = None
    if args.x_axis == "per_size" and args.show_mean_powerlaw:
        mean_pl = _mean_powerlaw_curve(runs)

    _apply_paper_rc()

    budgets = [b for b, _, _ in runs]
    span = max(budgets) / max(min(budgets), 1)
    if args.x_axis != "per_size":
        use_log_x = False
    elif args.log_x:
        use_log_x = True
    elif args.linear_x:
        use_log_x = False
    else:
        use_log_x = span >= 4.0

    # Slightly wider than single column for colorbar.
    fig_w = 6.8 if args.x_axis == "per_size" else 6.2
    fig, ax = plt.subplots(figsize=(fig_w, 3.55), layout="constrained")

    title = args.title

    if args.x_axis == "per_size":
        _plot_per_size_mode(ax, runs, log_x=use_log_x, title=title, mean_powerlaw=mean_pl)
    else:
        _plot_board_mode(ax, runs, title=title)

    fig.patch.set_facecolor("white")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), facecolor="white", edgecolor="none")
    if out.suffix.lower() == ".png" and not args.no_auto_pdf:
        pdf_path = out.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=300, facecolor="white", edgecolor="none")
        print(f"wrote {out} and {pdf_path}", flush=True)
    else:
        print(f"wrote {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
