import matplotlib.pyplot as plt

def duo_line_graph(
    x_range,
    train_losses,
    val_losses,
    title="loss curve",
    xlabel="epoch",
    ylabel="loss",
    save_path=None,
):
    x_vals = list(x_range)
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


def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names=("not pizza", "pizza"),
    save_path=None,
):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal lengths")

    size = len(class_names)
    matrix = [[0] * size for _ in range(size)]
    for actual, predicted in zip(true_labels, predicted_labels):
        matrix[actual][predicted] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(size), class_names)
    ax.set_yticks(range(size), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Validation confusion matrix")

    threshold = max(max(row) for row in matrix) / 2
    for row in range(size):
        for column in range(size):
            value = matrix[row][column]
            color = "white" if value > threshold else "black"
            ax.text(column, row, value, ha="center", va="center", color=color)

    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return matrix
    