import base64
import gc
import io

import matplotlib
import matplotlib.pyplot as plt

from zav.agents_sdk.adapters.async_wrapper import asyncify

matplotlib.use("Agg")

# Brand primary + highlight, then the design-system categorical chart palette,
# so charts match the product instead of matplotlib's defaults.
PLOT_PALETTE = [
    "#005d83",
    "#f17225",
    "#4f5ee8",
    "#38b6d3",
    "#9055dc",
    "#d4a627",
    "#ea6c39",
]

# Set once at import (process-global, before any rendering) — safe under the
# thread-pool used by asyncify, unlike per-call rcParams mutation.
matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Prefer a clean sans matching the product UI; falls back to DejaVu Sans
        # (matplotlib's default) when none of these are installed.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#1a1f29",
        "axes.labelsize": 11,
        "axes.labelcolor": "#3a3f4a",
        "axes.edgecolor": "#d0d4da",
        "xtick.color": "#3a3f4a",
        "ytick.color": "#3a3f4a",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
    }
)


def style_axes(ax) -> None:
    """Apply the modern, business-grade look to a single Axes (thread-safe)."""
    ax.set_prop_cycle(color=PLOT_PALETTE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e7e9ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


async def create_plot_image_uri(fig) -> str:
    try:
        buffer = io.BytesIO()
        await asyncify(fig.savefig)(
            buffer,
            format="png",
            bbox_inches="tight",
            dpi=100,
            facecolor="white",
        )
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        plt.close(fig)
        buffer.close()
        gc.collect()
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        plt.close(fig)
        raise e
