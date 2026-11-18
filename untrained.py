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
    print(user_type)
    # image_size = 64
    #not_pizza_path, pizza_path = dataset_path.main(image_size)

def valid_argv(argv) -> int:
    if len(argv) != 2:
        print("Wrong Usage!")
        print("Usage: python untrained.py <path to image(s)>")
        return -1

    user_path = Path(argv[1])

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

main()