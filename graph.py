import matplotlib.pyplot as plt

def line_graph(
    x_range,
    train_losses,
    val_losses,
    title="loss curve",
    xlabel="epoch",
    ylabel="loss",
    save_path=None,
):
    x_vals = list(x_range)
    if len(x_vals) != len(train_losses) or len(x_vals) != len(val_losses):
        raise ValueError("x_range, train_losses, and val_losses must have equal lengths")

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        x_vals, train_losses, color="tab:blue", linewidth=2,
        marker="o", markersize=5, label="training loss"
    )
    plt.plot(
        x_vals, val_losses, color="tab:orange", linewidth=2,
        marker="o", markersize=5, label="validation loss"
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    