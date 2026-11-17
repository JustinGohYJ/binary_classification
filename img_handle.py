import kagglehub
from PIL import Image
from pathlib import Path

img_size = 64

def main() -> None:
    dataset = kagglehub.dataset_download("carlosrunner/pizza-not-pizza") + '\\pizza_not_pizza\\'
    path_np, image_path_np, path_p, image_path_p = create_paths(dataset)
    # save resized img to new path
    resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size)
    

# resize all image from cur_path and saved it to new_path
def resize_img(cur_path, new_path, size) -> int:
    i = 0
    for item in cur_path.iterdir():
        if not valid_file(item) or not valid_image(item):
            continue

        # valid image
        with Image.open(item) as img:
            i += 1
            resized_img = img.resize((size, size), resample=(Image.Resampling.LANCZOS))
            resized_img.save(new_path / f"{i:03d}.png")

    return i

# just handles the printing
def resize_img_print(path_np, image_path_np, path_p, image_path_p, img_size) -> None:
    print("Handling non pizza files", end='\r')
    print(f"Total non pizza files: {resize_img(path_np, image_path_np, img_size)}")
    print("Handling pizza files", end='\r')
    print(f"Total pizza files: {resize_img(path_p, image_path_p, img_size)}")

# create path variables
def create_paths(path) -> tuple[Path, Path, Path, Path]:
    print("Handling path creation", end='\r')

    path_np = Path(path + 'not_pizza')
    path_p = Path(path + 'pizza')

    image_path_np = Path("not_pizza_resized")
    image_path_np.mkdir(exist_ok=True)

    image_path_p = Path("pizza_resized")
    image_path_p.mkdir(exist_ok=True)

    print("Path to original dataset files:", path)
    return path_np, image_path_np, path_p, image_path_p

# validate file
def valid_file(item) -> bool:
    if not item.is_file():
        print(f"Skipped {item.name}, not a file",)
        return False
    return True

# validate image
def valid_image(item) -> bool:
    try: 
        with Image.open(item) as img:
            img.verify()
    except (OSError, ValueError):
        print(f"Skipped {item.name}, non image")
        return False
    
    return True

if __name__ == "__main__":
    main()