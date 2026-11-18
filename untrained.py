import dataset_path
import sys
from pathlib import Path

def main() -> None:
    '''
    0: image
    1: folder
    -1: invalid
    '''
    user_type = valid_argv(sys.argv)
    if user_type == -1:
        return

    print(user_type)
    image_size = 64

    epoch = validate_user_num(int, 1, 9999, "Enter a suitable epochs")
    learning_rate = validate_user_num(float, 0.0001, 0.01, "Choose the learning rate")
    data_split = validate_user_datasplit()

    not_pizza_path, pizza_path = dataset_path.main(image_size)


def validate_user_num(type, min, max, message):
    while True:
        try:
            val = type(input(f"{message} [{min} - {max}]: "))

            if min <= val <= max:
                return val

            print(f"Input must be between {min} and {max}")
        except ValueError:
            print(f"Invalid input, expected {type.__name__}")


def validate_user_datasplit() -> list[int]:
    print("Input a suitable percentage split for the data [Must sum to 100]")
    print("Training, Validation, Test")

    while True:
        total = 100
        training = validate_user_num(int, 1, total - 2, "Training: ")
        total -= training
        validation = validate_user_num(int, 1, total - 1, "Validation: ")
        test = total - validation

        confirm = input("\nConfirm data split (y/n): \n"
                          f"Training: {training}\n"
                          f"Validation: {validation}\n"
                          f"Test: {test}\n")

        if confirm.lower() == 'y':
            return [training, validation, test]


def valid_argv(argv) -> int:
    if len(argv) != 2:
        print("Wrong Usage!")
        print("Usage: python untrained.py <path to image(s)>")
        return -1

    user_path = Path(argv[1])

    # checks valid dir or path
    if not dataset_path.valid_file(user_path):
        if not user_path.is_dir():
            print("Ensure <path> is a valid file or directory in the parent directory")
            return -1

        # valid folder
        return 1

    if not dataset_path.valid_image(user_path):
        print("Ensure <file> is a valid image")
        return -1

    # valid image
    return 0

if __name__ == "__main__":
    main()