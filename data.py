import kagglehub
import colour
import random
from PIL import Image
from pathlib import Path

def save_usable_dataset(img_size) -> tuple[Path, Path]:
    training_dataset = Path(kagglehub.dataset_download("carlosrunner/pizza-not-pizza"))
    training_dataset = training_dataset / "pizza_not_pizza"

    path_np, image_path_np, path_p, image_path_p = create_paths(training_dataset)
    # save resized img to new path
    non_pizza_amt, pizza_amt = resize_img_dir(path_np, image_path_np, path_p, image_path_p, img_size)
    
    return image_path_np, image_path_p, non_pizza_amt, pizza_amt


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


# handles the printing, return the amt for both 
def resize_img_dir(path_np, image_path_np, path_p, image_path_p, img_size) -> tuple[int, int]:
    print("Handling non pizza files", end='\r', flush=True)
    non_pizza_total = resize_img(path_np, image_path_np, img_size)
    print(f"{colour.BLUE}Total non pizza files: {non_pizza_total}{colour.RESET}")

    print("Handling pizza files", end='\r', flush=True)
    pizza_total = resize_img(path_p, image_path_p, img_size)
    print(f"{colour.BLUE}Total pizza files: {pizza_total}{colour.RESET}")

    return non_pizza_total, pizza_total


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


def valid_file(item) -> bool:
    if not item.is_file():
        return False
    return True


def valid_image(item) -> bool:
    try: 
        with Image.open(item) as img:
            img.verify()
    except (OSError, ValueError):
        return False
    
    return True


def build_samples(image_dirs):
    samples = []

    # label will just be 0 or 1 as image dirs only have 2 obj
    for label, dir in enumerate(image_dirs):
        for path in Path(dir).iterdir():
            samples.append((path, label))

    random.shuffle(samples)
    return samples


# split the data into train, val, test, remainder into trian
def build_split(samples, data_split):
    size = len(samples)
    n_test = size * data_split[2] // 100
    n_val = size * data_split[1] // 100
    n_train = size - n_test - n_val

    i_val = n_train + n_val
    return samples[:n_train], samples[n_train:i_val], samples[i_val:]