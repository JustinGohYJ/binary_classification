import kagglehub
from PIL import Image
from pathlib import Path

def main() -> None:
    # Prepare all paths
    path, path_np, image_path_np, path_p, image_path_p = create_paths()
    print("Path to dataset files:", path)

    # size of img, keep to power of 2
    def_size = 64
    print(f"Total non pizza files: {resize_img(path_np, image_path_np, def_size)}")
    print(f"Total pizza files: {resize_img(path_p, image_path_p, def_size)}")

# resize all image from cur_path and saved it to new_path
def resize_img(cur_path, new_path, size) -> int:
    i = 0
    for item in cur_path.iterdir():
        if not item.is_file():
            print(f"file not found: {item}")
            continue

        with Image.open(item) as img:
            i += 1
            resized_img = img.resize((size, size), resample=(Image.Resampling.LANCZOS))
            resized_img.save(new_path / f"{i}.png")

    return i

def create_paths() -> tuple[Path, Path, Path, Path, Path]:
    path = kagglehub.dataset_download("carlosrunner/pizza-not-pizza") + '\\pizza_not_pizza\\'
    path_np = Path(path + 'not_pizza')
    path_p = Path(path + 'pizza')

    image_path_np = Path("not_pizza_resized")
    image_path_np.mkdir(parents=True, exist_ok=True)

    image_path_p = Path("pizza_resized")
    image_path_p.mkdir(parents=True, exist_ok=True)
    return path, path_np, image_path_np, path_p, image_path_np

if __name__ == "__main__":
    main()