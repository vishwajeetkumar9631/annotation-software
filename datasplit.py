from __future__ import annotations

import argparse
import ast
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the annotation app's YOLO source dataset into train/valid/test folders."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("backend/data/source"),
        help="Folder containing images/, labels/, classes.txt, and optionally data.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/data/split"),
        help="Output folder for the split dataset.",
    )
    parser.add_argument("--train", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--valid", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable splits.")
    parser.add_argument(
        "--empty-labels",
        action="store_true",
        help="Create empty label files for images that do not have a matching label.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output folder before writing the new split.",
    )
    parser.add_argument(
        "--yolo-layout",
        action="store_true",
        help="Read and write YOLO layout folders like images/train and labels/train.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, valid_ratio, test_ratio) < 0:
        raise ValueError("Split ratios cannot be negative.")

    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    if abs(total - 1.0) > 0.000001:
        raise ValueError("Split ratios must add up to 1.0.")


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(
    images: list[Path],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = round(total * train_ratio)
    valid_count = round(total * valid_ratio)

    if train_count + valid_count > total:
        valid_count = max(0, total - train_count)

    train_end = train_count
    valid_end = train_end + valid_count

    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def class_names(source_dir: Path) -> list[str]:
    classes_path = source_dir / "classes.txt"
    if classes_path.exists():
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    data_yaml = source_dir / "data.yaml"
    if not data_yaml.exists():
        return []

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names_value = line.split(":", 1)[1].strip()
            try:
                names = ast.literal_eval(names_value)
            except (SyntaxError, ValueError):
                return []
            return [str(name) for name in names]

    return []


def copy_split(
    splits: dict[str, list[Path]],
    labels_dir: Path,
    output_dir: Path,
    create_empty_labels: bool,
    yolo_layout: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for split_name, paths in splits.items():
        if yolo_layout:
            split_images_dir = output_dir / "images" / split_name
            split_labels_dir = output_dir / "labels" / split_name
        else:
            split_images_dir = output_dir / split_name / "images"
            split_labels_dir = output_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in paths:
            shutil.copy2(image_path, split_images_dir / image_path.name)

            label_path = labels_dir / f"{image_path.stem}.txt"
            target_label_path = split_labels_dir / label_path.name
            if label_path.exists():
                shutil.copy2(label_path, target_label_path)
            elif create_empty_labels:
                target_label_path.write_text("", encoding="utf-8")

        counts[split_name] = len(paths)

    return counts


def write_data_yaml(output_dir: Path, names: list[str], yolo_layout: bool) -> None:
    names_text = repr(names)
    train_path = "images/train" if yolo_layout else "train/images"
    valid_path = "images/valid" if yolo_layout else "valid/images"
    test_path = "images/test" if yolo_layout else "test/images"
    data_yaml = (
        "path: .\n"
        f"train: {train_path}\n"
        f"val: {valid_path}\n"
        f"test: {test_path}\n"
        f"nc: {len(names)}\n"
        f"names: {names_text}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_dir / "classes.txt").write_text("\n".join(names), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.valid, args.test)

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    if args.yolo_layout:
        images_dir = source_dir / "images" / "train"
        labels_dir = source_dir / "labels" / "train"
    else:
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    images = image_paths(images_dir)
    if not images:
        raise ValueError(f"No supported image files found in: {images_dir}")

    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output folder must be outside the source folder to avoid mixing datasets.")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_images(images, args.train, args.valid, args.seed)
    counts = copy_split(splits, labels_dir, output_dir, args.empty_labels, args.yolo_layout)
    names = class_names(source_dir)
    write_data_yaml(output_dir, names, args.yolo_layout)

    print(f"Dataset split written to: {output_dir}")
    print(f"Train: {counts['train']} image(s)")
    print(f"Valid: {counts['valid']} image(s)")
    print(f"Test: {counts['test']} image(s)")
    print(f"Classes: {len(names)}")


if __name__ == "__main__":
    main()
