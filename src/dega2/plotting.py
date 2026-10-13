"""Convergence plotting (matplotlib is an optional dependency)."""
import numpy as np


def plot_convergence(result, show=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required to plot the convergence curve: "
            "pip install \"dega2[plot]\""
        ) from exc

    y = np.ravel(result.history)
    t = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, y, color="#d62728", linewidth=2, label="min φ")
    ax.fill_between(t, y, y.max(), color="#d62728", alpha=0.08)
    ax.scatter([t[-1]], [y[-1]], color="#d62728", zorder=3)
    ax.annotate(f"φ* = {y[-1]:.4g}", xy=(t[-1], y[-1]),
                xytext=(-90, 25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("min of  phi")
    ax.set_title("Convergence of DEGA-2")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig
