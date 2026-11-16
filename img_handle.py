import kagglehub
from PIL import Image
from pathlib import Path

img_size = 64

def main() -> None:
    #prepare all path
    path_np, image_path_np, path_p, image_path_p = create_paths()

    # save resized img to new path
    resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size)
    

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
            resized_img.save(new_path / f"{i:03d}.png")

    return i

# just handle the printing
def resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size) -> None:
    print("Handling non pizza files", end='\r')
    print(f"Total non pizza files: {resize_img(path_np, image_path_np, img_size)}")
    print("Handling pizza files", end='\r')
    print(f"Total pizza files: {resize_img(path_p, image_path_p, img_size)}")

# download kaggle dataset, and create path
def create_paths() -> tuple[Path, Path, Path, Path]:
    print("Handling path creation", end='\r')
    path = kagglehub.dataset_download("carlosrunner/pizza-not-pizza") + '\\pizza_not_pizza\\'
    path_np = Path(path + 'not_pizza')
    path_p = Path(path + 'pizza')

    image_path_np = Path("not_pizza_resized")
    image_path_np.mkdir(exist_ok=True)

    image_path_p = Path("pizza_resized")
    image_path_p.mkdir(exist_ok=True)

    print("Path to dataset files:", path)
    return path_np, image_path_np, path_p, image_path_p

if __name__ == "__main__":
    main()