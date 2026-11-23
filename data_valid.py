import kagglehub
import colour
from PIL import Image
from pathlib import Path

def main(img_size) -> tuple[Path, Path]:
    training_dataset = Path(kagglehub.dataset_download("carlosrunner/pizza-not-pizza"))
    training_dataset = training_dataset / "pizza_not_pizza"
    path_np, image_path_np, path_p, image_path_p = create_paths(training_dataset)
    # save resized img to new path
    resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size)
    
    return image_path_np, image_path_p


# create path variables
def create_paths(path) -> tuple[Path, Path, Path, Path]:
    print("Handling path creation", end='\r')

    path_np = path / "not_pizza"
    path_p = path / "pizza"

    image_path_np = Path("not_pizza_resized")
    image_path_np.mkdir(exist_ok=True)

    image_path_p = Path("pizza_resized")
    image_path_p.mkdir(exist_ok=True)

    print("Path to original dataset files:", path)
    return path_np, image_path_np, path_p, image_path_p


# just handles the printing
def resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size) -> None:
    print("Handling non pizza files", end='\r', flush=True)
    print(f"{colour.BLUE}Total non pizza files: {resize_img(path_np, image_path_np, img_size)}{colour.RESET}")
    print("Handling pizza files", end='\r', flush=True)
    print(f"{colour.BLUE}Total pizza files: {resize_img(path_p, image_path_p, img_size)}{colour.RESET}")


# resize all image from cur_path and saved it to new_path
def resize_img(cur_path, new_path, size) -> int:
    i = 0
    for item in cur_path.iterdir():
        if not valid_file(item):
            print(f"Skipped {item.name}, not a file",)
            continue

        if not valid_image(item):
            print(f"Skipped {item.name}, non image")
            continue

        # valid image
        with Image.open(item) as img:
            i += 1
            resized_img = img.resize((size, size), resample=(Image.Resampling.LANCZOS))
            resized_img.save(new_path / f"{i:03d}.png")

    return i


# validate file
def valid_file(item) -> bool:
    if not item.is_file():
        return False
    return True


# validate image
def valid_image(item) -> bool:
    try: 
        with Image.open(item) as img:
            img.verify()
    except (OSError, ValueError):
        return False
    
    return True