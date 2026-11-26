import data
import colour
import sys
import mlp
import graph
import numpy as np
from pathlib import Path

def main() -> None:
    '''
    0: img
    1: dir
    -1: invalid
    '''
    # user_type = valid_argv(sys.argv)
    # if user_type == -1:
    #     return

    # print(user_type)
    # epoch = validate_user_num(int, 1, 999, "Enter a suitable epochs")
    # learning_rate = validate_user_num(float, 0.0001, 1, "Choose the learning rate")
    data_split = validate_user_datasplit()

    not_pizza_path = Path("not_pizza_resized")
    pizza_path = Path("pizza_resized")
    
    samples = data.build_samples([not_pizza_path, pizza_path])
    train, val, test = data.build_split(samples, data_split)
    epoch = 30
    batch_size = 32
    W, B = mlp.init_network(12288, 256, 100, 1)
    optimizer = mlp.Momentum(W, B, beta=0.9)
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_W = None
    best_B = None
    for i in range(epoch):
        # train
        W, B = mlp.full_pass(
            train, W, B, 0.001, batch_size, optimizer
        )
        # to be fair we judge it after updates
        avg_train_loss, train_accuracy = mlp.evaluate(train, W, B)
        train_losses.append(avg_train_loss)

        # val
        avg_val_loss, val_accuracy = mlp.evaluate(val, W, B)
        val_losses.append(avg_val_loss)

        # save best parameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = i + 1
            best_W = [layer.copy() for layer in W]
            best_B = [layer.copy() for layer in B]


        print(f"Epoch: {i + 1}")
        print(f"Training Loss: {avg_train_loss:.3f}")
        print(f"Training Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss: {avg_val_loss:.3f}")
        print(f"Validation Accuracy: {val_accuracy:.2%}")

    W, B = best_W, best_B
    print(f"Best epoch: {best_epoch} (validation loss: {best_val_loss:.3f})")

    graph.line_graph(range(epoch), train_losses, val_losses, save_path="bestweigth33.png")

    # confusion matrix on best weights
    true_labels, predicted_labels = mlp.obtain_predictions(test, W, B)
    graph.plot_confusion_matrix(
        true_labels,
        predicted_labels,
        save_path="bestweigh33.png",
    )


# validate ranged user input for int/float
def validate_user_num(type, min, max, message):
    print("")
    while True:
        try:
            val = type(input(f"{message} [{min} - {max}]: "))
        except ValueError:
            print(f"{colour.RED}Invalid input, expected {colour.YELLOW}{type.__name__}{colour.RESET}")
            continue

        if not min <= val <= max:
            print(f"{colour.RED}Input must be between {colour.YELLOW}{min} and {max}{colour.RESET}")
            continue

        return val
            

# handles datasplit input returns percentage <1
def validate_user_datasplit() -> list[int]:
    print("Input a suitable percentage split for the data [Must sum to 100]")
    print(f"{colour.PURPLE}Training, Validation, Test{colour.RESET}")

    while True:
        # Force at least 1% in each category
        total = 100
        training = validate_user_num(int, 1, total - 2, "Training: ")
        total -= training
        validation = validate_user_num(int, 1, total - 1, "Validation: ")
        test = total - validation

        confirm = input(f"\nConfirm data split {colour.YELLOW}(y/n){colour.RESET}: \n"
                          f"Training: {training}\n"
                          f"Validation: {validation}\n"
                          f"Test: {test}\n"
                          f"{colour.YELLOW}Answer: {colour.RESET}")

        # defaults no
        if confirm.lower() == 'y':
            return [training, validation, test]


def valid_argv(argv) -> int:

    # missing / extra argv
    if len(argv) != 2:
        print(f"{colour.RED}Wrong Usage!{colour.RESET}")
        print(f"Usage: python untrained.py {colour.YELLOW}<path to image(s)>{colour.RESET}")
        return -1

    user_path = Path(argv[1])

    # checks valid dir
    if not data.valid_file(user_path):
        if not user_path.is_dir():
            print(f"{colour.RED}Ensure <path> is a valid file or directory in the parent directory{colour.RESET}")
            return -1

        # valid folder
        return 1

    # valid file, invalid image
    if not data.valid_image(user_path):
        print(f"{colour.RED}Ensure <file> is a valid image{colour.RESET}")
        return -1

    # valid image
    return 0

if __name__ == "__main__":
    main()