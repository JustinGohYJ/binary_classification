import data_valid
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
    image_size = 64

    # epoch = validate_user_num(int, 1, 9999, "Enter a suitable epochs")
    # learning_rate = validate_user_num(float, 0.0001, 1, "Choose the learning rate")
    # data_split = validate_user_datasplit()

    not_pizza_path = Path("not_pizza_resized")
    pizza_path = Path("pizza_resized")

    epoch = 1
    batch_size = 32
    input_size = image_size * image_size * 3
    W, B = mlp.init_network(input_size, 4096, 512, 1)
    losses = []
    for i in range(epoch):
        W, B, avg_loss = mlp.train_mini_batch(
            [not_pizza_path, pizza_path], W, B, 0.003, batch_size
        )
        losses.append(avg_loss)
        print("\r         \r", end="")
        print(epoch - i)

    graph.line_graph(range(epoch), losses)

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
            

# handles datasplit input
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
    if not data_valid.valid_file(user_path):
        if not user_path.is_dir():
            print(f"{colour.RED}Ensure <path> is a valid file or directory in the parent directory{colour.RESET}")
            return -1

        # valid folder
        return 1

    # valid file, invalid image
    if not data_valid.valid_image(user_path):
        print(f"{colour.RED}Ensure <file> is a valid image{colour.RESET}")
        return -1

    # valid image
    return 0

if __name__ == "__main__":
    main()