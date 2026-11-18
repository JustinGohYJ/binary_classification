import dataset_path
import sys
from pathlib import Path

def main() -> None:
    user_type = "img"
    if len(sys.argv) != 2:
        print("Wrong Usage!")
        print("Usage: python untrained.py <path to image(s)>")
        return

    user_path = Path(sys.argv[1])

    if not dataset_path.valid_file(user_path):
        if not user_path.is_dir():
            print("Ensure <path> is a valid file or directory in the parent directory")
            return

        user_type = "dir"

    if not dataset_path.valid_image(user_path):
        print("Ensure <image> is a valid image")
        return


    image_size = 64
    not_pizza_path, pizza_path = dataset_path.main(image_size)

def valid_argv(path) -> bool:
    return True

main()