"""
common.plotting — small shared matplotlib helpers used by the figure script
and the generator's validation panels.

The many near-identical ``plabel`` variants that used to be re-declared in each
figure block are consolidated into a single helper. If a specific figure needs
a non-default position, pass ``x``/``y`` explicitly.
"""

import matplotlib.pyplot as plt

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
    from IPython.display import display
except Exception:
    def display(obj):  # noqa: E501
        pass  # no-op outside notebooks


def plabel(ax, lbl, x=-0.16, y=1.10, fontsize=14, color="#1c1917"):
    """Draw a bold 'A'/'B'/... panel label in axes coordinates."""
    ax.text(x, y, lbl, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", color=color)


def panel_label(ax, letter, dx=-0.12, dy=1.06):
    """Left-aligned variant of :func:`plabel` (used by Figure 1 panels)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left")


def save_and_show(fig, filepath):
    """Save a figure to disk and display it if running inside Jupyter."""
    fig.savefig(filepath, bbox_inches="tight")
    display(fig)
    plt.show()
    plt.close(fig)
    print(f"  Saved: {filepath}")


__all__ = ["plabel", "panel_label", "save_and_show"]