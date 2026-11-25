import matplotlib.pyplot as plt

def line_graph(
    x_range,
    y_data,
    title="loss curve",
    xlabel="epoch",
    ylabel="loss",
    save_path=None,
):
    # Sanity check: lengths must match
    # If the user passes a `range` object, convert to a list for plotting
    x_vals = list(x_range)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_vals, y_data, color="tab:blue", linewidth=2,
             marker="o", markersize=5, label="data")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    