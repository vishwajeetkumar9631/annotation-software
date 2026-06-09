from __future__ import annotations

import io
import json
import math
import os
import shutil
import stat
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SOURCE_DIR = DATA_DIR / "source"
SOURCE_IMAGE_DIR = SOURCE_DIR / "images"
SOURCE_LABEL_DIR = SOURCE_DIR / "labels"
SOURCE_DATA_YAML = SOURCE_DIR / "data.yaml"
FRONTEND_DATA_YAML = ROOT.parent / "frontend" / "data.yaml"
DB_PATH = DATA_DIR / "db.json"
MODEL_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"
TRAINING_DATASET_DIR = DATA_DIR / "training"
BEST_MODEL_PATH = MODEL_DIR / "best.pt"
STORE_LOCK = threading.Lock()
TRAINING_LOCK = threading.Lock()
TRAINING_STATUS: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "message": "No training has started.",
    "model_path": None,
    "error": None,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
LABEL_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#ca8a04",
    "#9333ea",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#ea580c",
    "#0f766e",
    "#7c3aed",
    "#be123c",
    "#0284c7",
    "#4d7c0f",
    "#a16207",
    "#c026d3",
]


class Label(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    color: str = "#2563eb"


class LabelUpdate(BaseModel):
    name: str
    color: str = "#2563eb"


class FileRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str
    width: int | None = None
    height: int | None = None


class FolderImportPayload(BaseModel):
    folder_path: str
    recursive: bool = False
    replace_existing: bool = True


class BoundingBox(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_id: str
    label_id: str
    type: Literal["bbox"] = "bbox"
    x: float
    y: float
    width: float
    height: float


class AnnotationPayload(BaseModel):
    annotations: list[dict[str, Any]]


class FileMetadata(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ImageResizePayload(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RotationAugmentPayload(BaseModel):
    degrees: list[int]
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class YoloTrainPayload(BaseModel):
    base_model: str = "yolov8n.pt"
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=640, ge=128, le=2048)
    batch_size: int = Field(default=4, ge=1, le=128)
    device: Literal["auto", "gpu", "cpu"] = "gpu"
    workers: int = Field(default=0, ge=0, le=32)


class AutoAnnotatePayload(BaseModel):
    file_ids: list[str] | None = None
    confidence: float = Field(default=0.35, ge=0.01, le=1.0)
    replace_existing: bool = False
    device: Literal["auto", "gpu", "cpu"] = "auto"


class Store(BaseModel):
    labels: list[Label] = Field(default_factory=list)
    files: list[FileRecord] = Field(default_factory=list)
    annotations: list[BoundingBox] = Field(default_factory=list)


def ensure_store() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SOURCE_DIR.mkdir(exist_ok=True)
    SOURCE_IMAGE_DIR.mkdir(exist_ok=True)
    SOURCE_LABEL_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    TRAINING_DATASET_DIR.mkdir(exist_ok=True)
    if not DB_PATH.exists():
        DB_PATH.write_text(Store().model_dump_json(indent=2), encoding="utf-8")


def read_store() -> Store:
    ensure_store()
    return Store.model_validate_json(DB_PATH.read_text(encoding="utf-8"))


def write_store(store: Store) -> None:
    ensure_store()
    temp_path = DB_PATH.with_suffix(".json.tmp")
    temp_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    temp_path.replace(DB_PATH)


def image_metadata(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lower().lstrip(".")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image file: {path.name}") from exc

    content_type = Image.MIME.get(image_format.upper(), f"image/{path.suffix.lower().lstrip('.')}")
    return width, height, content_type


def repair_duplicate_label_colors(store: Store) -> bool:
    used_colors: set[str] = set()
    changed = False

    for index, label in enumerate(store.labels):
        color = label.color.lower()
        if color not in used_colors:
            used_colors.add(color)
            continue

        for candidate in LABEL_COLORS:
            if candidate.lower() not in used_colors:
                label.color = candidate
                used_colors.add(candidate.lower())
                changed = True
                break
        else:
            fallback = LABEL_COLORS[index % len(LABEL_COLORS)]
            label.color = fallback
            used_colors.add(fallback.lower())
            changed = True

    return changed


def retry_readonly_delete(function: Any, path: str, _exc_info: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def clear_directory(directory: Path, *, ignore_errors: bool = False) -> None:
    if not directory.exists():
        return
    for item in directory.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, onerror=retry_readonly_delete)
            else:
                item.unlink()
        except PermissionError:
            if not ignore_errors:
                raise


def uploaded_path(file_id: str) -> Path | None:
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return None
    return matches[0]


def remove_missing_file_records(store: Store) -> bool:
    existing_file_ids = {file.id for file in store.files if uploaded_path(file.id) is not None}
    if len(existing_file_ids) == len(store.files):
        return False

    store.files = [file for file in store.files if file.id in existing_file_ids]
    store.annotations = [annotation for annotation in store.annotations if annotation.file_id in existing_file_ids]
    return True


def write_current_data_yaml(store: Store) -> None:
    ensure_store()
    (SOURCE_DIR / "classes.txt").write_text("\n".join(label.name for label in store.labels), encoding="utf-8")
    yaml_content = (
        f"path: {json.dumps(SOURCE_DIR.resolve().as_posix())}\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(store.labels)}\n"
        f"names: {json.dumps([label.name for label in store.labels])}\n"
    )
    SOURCE_DATA_YAML.write_text(yaml_content, encoding="utf-8")
    if FRONTEND_DATA_YAML.parent.exists():
        FRONTEND_DATA_YAML.write_text(yaml_content, encoding="utf-8")


def yolo_rows(file: FileRecord, annotations: list[BoundingBox], label_index: dict[str, int]) -> list[str]:
    image_width = file.width or 1
    image_height = file.height or 1
    rows: list[str] = []

    for annotation in annotations:
        if annotation.label_id not in label_index:
            continue
        center_x = (annotation.x + annotation.width / 2) / image_width
        center_y = (annotation.y + annotation.height / 2) / image_height
        width = annotation.width / image_width
        height = annotation.height / image_height
        rows.append(f"{label_index[annotation.label_id]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    return rows


def yolo_seg_rows(file: FileRecord, annotations: list[BoundingBox], label_index: dict[str, int]) -> list[str]:
    image_width = file.width or 1
    image_height = file.height or 1
    rows: list[str] = []

    for annotation in annotations:
        if annotation.label_id not in label_index:
            continue
        x1 = annotation.x / image_width
        y1 = annotation.y / image_height
        x2 = (annotation.x + annotation.width) / image_width
        y2 = annotation.y / image_height
        x3 = (annotation.x + annotation.width) / image_width
        y3 = (annotation.y + annotation.height) / image_height
        x4 = annotation.x / image_width
        y4 = (annotation.y + annotation.height) / image_height
        rows.append(
            f"{label_index[annotation.label_id]} "
            f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
            f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
        )

    return rows


def unique_dataset_filename(filename: str, used_names: set[str]) -> str:
    path = Path(filename)
    suffix = path.suffix
    stem = path.stem or "image"
    candidate = f"{stem}{suffix}"
    counter = 2

    while candidate.lower() in used_names or Path(candidate).stem.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate.lower())
    used_names.add(Path(candidate).stem.lower())
    return candidate


def build_yolo_zip(store: Store) -> bytes:
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    annotations_by_file = {
        file.id: [item for item in store.annotations if item.file_id == file.id]
        for file in store.files
    }

    archive = io.BytesIO()
    used_image_names: set[str] = set()

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("classes.txt", "\n".join(label.name for label in store.labels))
        zip_file.writestr(
            "data.yaml",
            "path: .\n"
            "train: images/train\n"
            "val: images/train\n"
            f"nc: {len(store.labels)}\n"
            f"names: {[label.name for label in store.labels]!r}\n",
        )

        for file in store.files:
            source = uploaded_path(file.id)
            if source is None:
                continue

            image_name = unique_dataset_filename(file.filename, used_image_names)
            label_name = f"{Path(image_name).stem}.txt"
            rows = yolo_rows(file, annotations_by_file.get(file.id, []), label_index)

            zip_file.write(source, f"images/train/{image_name}")
            zip_file.writestr(f"labels/train/{label_name}", "\n".join(rows))

    return archive.getvalue()


def build_yolo_seg_zip(store: Store) -> bytes:
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    annotations_by_file = {
        file.id: [item for item in store.annotations if item.file_id == file.id]
        for file in store.files
    }

    archive = io.BytesIO()
    used_image_names: set[str] = set()

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("classes.txt", "\n".join(label.name for label in store.labels))
        zip_file.writestr(
            "data.yaml",
            "path: .\n"
            "train: images/train\n"
            "val: images/train\n"
            f"nc: {len(store.labels)}\n"
            f"names: {[label.name for label in store.labels]!r}\n",
        )

        for file in store.files:
            source = uploaded_path(file.id)
            if source is None:
                continue

            image_name = unique_dataset_filename(file.filename, used_image_names)
            label_name = f"{Path(image_name).stem}.txt"
            rows = yolo_seg_rows(file, annotations_by_file.get(file.id, []), label_index)

            zip_file.write(source, f"images/train/{image_name}")
            zip_file.writestr(f"labels/train/{label_name}", "\n".join(rows))

    return archive.getvalue()


def build_yolo_label_map(store: Store) -> dict[str, str]:
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    used_image_names: set[str] = set()
    output: dict[str, str] = {}

    for file in store.files:
        image_name = unique_dataset_filename(file.filename, used_image_names)
        rows = yolo_rows(file, [item for item in store.annotations if item.file_id == file.id], label_index)
        output[f"{Path(image_name).stem}.txt"] = "\n".join(rows)

    return output


def build_yolo_seg_label_map(store: Store) -> dict[str, str]:
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    used_image_names: set[str] = set()
    output: dict[str, str] = {}

    for file in store.files:
        image_name = unique_dataset_filename(file.filename, used_image_names)
        rows = yolo_seg_rows(file, [item for item in store.annotations if item.file_id == file.id], label_index)
        output[f"{Path(image_name).stem}.txt"] = "\n".join(rows)

    return output


def write_yolo_source_folder(store: Store) -> dict[str, int | str]:
    ensure_store()
    clear_directory(SOURCE_IMAGE_DIR)
    clear_directory(SOURCE_LABEL_DIR)
    ensure_store()
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    annotations_by_file = {
        file.id: [item for item in store.annotations if item.file_id == file.id]
        for file in store.files
    }
    used_image_names: set[str] = set()
    image_count = 0
    label_count = 0

    for file in store.files:
        source = uploaded_path(file.id)
        if source is None:
            continue

        image_name = unique_dataset_filename(file.filename, used_image_names)
        image_target = SOURCE_IMAGE_DIR / image_name
        label_target = SOURCE_LABEL_DIR / f"{Path(image_name).stem}.txt"
        rows = yolo_rows(file, annotations_by_file.get(file.id, []), label_index)

        shutil.copy2(source, image_target)
        label_target.write_text("\n".join(rows), encoding="utf-8")
        image_count += 1
        label_count += 1

    (SOURCE_DIR / "classes.txt").write_text("\n".join(label.name for label in store.labels), encoding="utf-8")
    write_current_data_yaml(store)

    return {
        "folder": str(SOURCE_DIR),
        "image_folder": str(SOURCE_IMAGE_DIR),
        "label_folder": str(SOURCE_LABEL_DIR),
        "images_written": image_count,
        "labels_written": label_count,
    }


def write_yolo_seg_source_folder(store: Store) -> dict[str, int | str]:
    ensure_store()
    clear_directory(SOURCE_IMAGE_DIR)
    clear_directory(SOURCE_LABEL_DIR)
    ensure_store()
    label_index = {label.id: index for index, label in enumerate(store.labels)}
    annotations_by_file = {
        file.id: [item for item in store.annotations if item.file_id == file.id]
        for file in store.files
    }
    used_image_names: set[str] = set()
    image_count = 0
    label_count = 0

    for file in store.files:
        source = uploaded_path(file.id)
        if source is None:
            continue

        image_name = unique_dataset_filename(file.filename, used_image_names)
        image_target = SOURCE_IMAGE_DIR / image_name
        label_target = SOURCE_LABEL_DIR / f"{Path(image_name).stem}.txt"
        rows = yolo_seg_rows(file, annotations_by_file.get(file.id, []), label_index)

        shutil.copy2(source, image_target)
        label_target.write_text("\n".join(rows), encoding="utf-8")
        image_count += 1
        label_count += 1

    (SOURCE_DIR / "classes.txt").write_text("\n".join(label.name for label in store.labels), encoding="utf-8")
    write_current_data_yaml(store)

    return {
        "folder": str(SOURCE_DIR),
        "image_folder": str(SOURCE_IMAGE_DIR),
        "label_folder": str(SOURCE_LABEL_DIR),
        "images_written": image_count,
        "labels_written": label_count,
    }


def load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Ultralytics is not installed. Run `pip install -r backend/requirements.txt` in the backend environment.",
        ) from exc
    return YOLO


def gpu_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
    }


def resolve_yolo_device(requested_device: str) -> str | int:
    environment = gpu_environment()
    if requested_device == "cpu":
        return "cpu"
    if environment["cuda_available"]:
        return 0
    if requested_device == "gpu":
        raise RuntimeError(
            "GPU training was requested, but CUDA-enabled PyTorch is not available. "
            f"Installed torch: {environment['torch_version']}. Install the CUDA PyTorch build in backend/.venv."
        )
    return "cpu"


def update_training_status(**values: Any) -> None:
    with TRAINING_LOCK:
        TRAINING_STATUS.update(values)


def training_dataset_summary(store: Store) -> dict[str, int]:
    annotated_file_ids = {annotation.file_id for annotation in store.annotations}
    return {
        "labels": len(store.labels),
        "files": len(store.files),
        "annotations": len(store.annotations),
        "annotated_files": len(annotated_file_ids),
    }


def write_yolo_training_dataset(store: Store) -> Path:
    annotated_file_ids = {annotation.file_id for annotation in store.annotations}
    annotated_files = sorted(
        [file for file in store.files if file.id in annotated_file_ids],
        key=lambda file: (file.filename.lower(), file.id),
    )
    if not annotated_files:
        raise RuntimeError("Save annotations before training.")

    clear_directory(TRAINING_DATASET_DIR)
    for split in ("train", "val"):
        (TRAINING_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TRAINING_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    if len(annotated_files) == 1:
        split_files = {"train": annotated_files, "val": annotated_files}
    else:
        train_count = max(1, min(len(annotated_files) - 1, round(len(annotated_files) * 0.8)))
        split_files = {
            "train": annotated_files[:train_count],
            "val": annotated_files[train_count:],
        }

    label_index = {label.id: index for index, label in enumerate(store.labels)}
    annotations_by_file = {
        file.id: [annotation for annotation in store.annotations if annotation.file_id == file.id]
        for file in annotated_files
    }
    used_names: set[str] = set()
    for split, files in split_files.items():
        for file in files:
            source = uploaded_path(file.id)
            if source is None:
                continue
            image_name = unique_dataset_filename(file.filename, used_names)
            shutil.copy2(source, TRAINING_DATASET_DIR / "images" / split / image_name)
            rows = yolo_rows(file, annotations_by_file[file.id], label_index)
            (TRAINING_DATASET_DIR / "labels" / split / f"{Path(image_name).stem}.txt").write_text(
                "\n".join(rows),
                encoding="utf-8",
            )

    data_yaml = TRAINING_DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {json.dumps(TRAINING_DATASET_DIR.resolve().as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(store.labels)}\n"
        f"names: {json.dumps([label.name for label in store.labels])}\n",
        encoding="utf-8",
    )
    return data_yaml


def run_yolo_training(payload: YoloTrainPayload) -> None:
    try:
        update_training_status(
            running=True,
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            message="Preparing YOLO dataset.",
            error=None,
        )
        with STORE_LOCK:
            store = read_store()
            summary = training_dataset_summary(store)
            if summary["labels"] == 0:
                raise RuntimeError("Create at least one label before training.")
            if summary["annotations"] == 0:
                raise RuntimeError("Save annotations before training.")
            training_data_yaml = write_yolo_training_dataset(store)

        update_training_status(message="Training YOLO model.")
        YOLO = load_yolo_class()
        device = resolve_yolo_device(payload.device)
        update_training_status(
            message=f"Training YOLO model on {'GPU 0' if device == 0 else 'CPU'}.",
            device=device,
            gpu_environment=gpu_environment(),
        )
        model = YOLO(payload.base_model)
        results = model.train(
            data=str(training_data_yaml),
            epochs=payload.epochs,
            imgsz=payload.image_size,
            batch=payload.batch_size,
            device=device,
            workers=payload.workers,
            amp=device != "cpu",
            cache=False,
            project=str(RUNS_DIR),
            name="detect",
            exist_ok=True,
        )

        save_dir = Path(str(getattr(results, "save_dir", RUNS_DIR / "detect")))
        best_model = save_dir / "weights" / "best.pt"
        if not best_model.exists():
            best_model = RUNS_DIR / "detect" / "weights" / "best.pt"
        if not best_model.exists():
            raise RuntimeError("Training finished, but best.pt was not found.")

        ensure_store()
        shutil.copy2(best_model, BEST_MODEL_PATH)
        update_training_status(
            running=False,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            message="Training complete.",
            model_path=str(BEST_MODEL_PATH),
            error=None,
        )
    except Exception as exc:
        update_training_status(
            running=False,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            message="Training failed.",
            error=str(exc),
        )


def predicted_boxes_for_file(
    model: Any,
    file: FileRecord,
    source: Path,
    labels: list[Label],
    confidence: float,
    device: str | int,
) -> list[BoundingBox]:
    results = model.predict(str(source), conf=confidence, device=device, verbose=False)
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    annotations: list[BoundingBox] = []
    xyxy_values = boxes.xyxy.cpu().tolist()
    class_values = boxes.cls.cpu().tolist()
    confidence_values = boxes.conf.cpu().tolist()
    image_width = float(file.width or 0)
    image_height = float(file.height or 0)

    for xyxy, class_index, detected_confidence in zip(xyxy_values, class_values, confidence_values):
        if detected_confidence < confidence:
            continue
        label_index = int(class_index)
        if label_index < 0 or label_index >= len(labels):
            continue

        x1, y1, x2, y2 = [float(value) for value in xyxy]
        left = max(0.0, x1)
        top = max(0.0, y1)
        right = min(image_width, x2) if image_width > 0 else x2
        bottom = min(image_height, y2) if image_height > 0 else y2
        width = round(right - left, 2)
        height = round(bottom - top, 2)
        if width <= 0 or height <= 0:
            continue

        annotations.append(
            BoundingBox(
                file_id=file.id,
                label_id=labels[label_index].id,
                type="bbox",
                x=round(left, 2),
                y=round(top, 2),
                width=width,
                height=height,
            )
        )
    return annotations


def rotated_bbox(
    annotation: BoundingBox,
    image_width: int,
    image_height: int,
    rotated_width: int,
    rotated_height: int,
    degrees: int,
) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)
    center_x = image_width / 2
    center_y = image_height / 2

    corners = [
        (annotation.x, annotation.y),
        (annotation.x + annotation.width, annotation.y),
        (annotation.x + annotation.width, annotation.y + annotation.height),
        (annotation.x, annotation.y + annotation.height),
    ]

    rotated_center_x = rotated_width / 2
    rotated_center_y = rotated_height / 2

    rotated_points = []
    for x, y in corners:
        offset_x = x - center_x
        offset_y = y - center_y
        rotated_x = cos_value * offset_x + sin_value * offset_y + rotated_center_x
        rotated_y = -sin_value * offset_x + cos_value * offset_y + rotated_center_y
        rotated_points.append((rotated_x, rotated_y))

    xs = [point[0] for point in rotated_points]
    ys = [point[1] for point in rotated_points]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(float(rotated_width), max(xs))
    bottom = min(float(rotated_height), max(ys))
    return round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2)


app = FastAPI(title="Annotation MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def state() -> Store:
    with STORE_LOCK:
        store = read_store()
        changed = remove_missing_file_records(store)
        changed = repair_duplicate_label_colors(store) or changed
        if changed:
            write_store(store)
    return store


@app.post("/api/labels")
def create_label(label: Label) -> Label:
    store = read_store()
    if any(existing.name.lower() == label.name.lower() for existing in store.labels):
        raise HTTPException(status_code=409, detail="Label already exists")
    store.labels.append(label)
    write_store(store)
    write_current_data_yaml(store)
    return label


@app.patch("/api/labels/{label_id}")
def update_label(label_id: str, payload: LabelUpdate) -> Label:
    store = read_store()
    label = next((item for item in store.labels if item.id == label_id), None)
    if label is None:
        raise HTTPException(status_code=404, detail="Label not found")

    cleaned_name = payload.name.strip()
    if not cleaned_name:
        raise HTTPException(status_code=400, detail="Label name is required")
    if any(item.id != label_id and item.name.lower() == cleaned_name.lower() for item in store.labels):
        raise HTTPException(status_code=409, detail="Label already exists")

    label.name = cleaned_name
    label.color = payload.color
    write_store(store)
    write_current_data_yaml(store)
    return label


@app.delete("/api/labels/{label_id}")
def delete_label(label_id: str) -> dict[str, int | str]:
    store = read_store()
    if not any(item.id == label_id for item in store.labels):
        raise HTTPException(status_code=404, detail="Label not found")

    annotation_count = sum(1 for item in store.annotations if item.label_id == label_id)
    store.labels = [item for item in store.labels if item.id != label_id]
    store.annotations = [item for item in store.annotations if item.label_id != label_id]
    write_store(store)
    write_current_data_yaml(store)
    return {"status": "deleted", "annotations_removed": annotation_count}


@app.post("/api/files")
def upload_file(file: UploadFile = File(...)) -> FileRecord:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image uploads are supported in this MVP")

    record = FileRecord(filename=Path(file.filename or f"upload{suffix}").name, content_type=file.content_type or "image/*")
    target = UPLOAD_DIR / f"{record.id}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        width, height, detected_content_type = image_metadata(target)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    record.width = width
    record.height = height
    record.content_type = file.content_type or detected_content_type

    with STORE_LOCK:
        store = read_store()
        store.files.append(record)
        write_store(store)
    return record


@app.post("/api/folders/import")
def import_folder(payload: FolderImportPayload) -> list[FileRecord]:
    folder = Path(payload.folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder path does not exist or is not a directory")

    paths = folder.rglob("*") if payload.recursive else folder.iterdir()
    image_paths = [path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise HTTPException(status_code=400, detail="No supported image files found in the folder")

    readable_images: list[tuple[Path, int, int, str]] = []
    for image_path in sorted(image_paths):
        try:
            width, height, detected_content_type = image_metadata(image_path)
        except HTTPException:
            continue
        readable_images.append((image_path, width, height, detected_content_type))

    if not readable_images:
        raise HTTPException(status_code=400, detail="No readable image files found in the folder")

    records: list[FileRecord] = []
    with STORE_LOCK:
        store = read_store()
        if payload.replace_existing:
            clear_directory(UPLOAD_DIR)
            clear_directory(SOURCE_IMAGE_DIR, ignore_errors=True)
            clear_directory(SOURCE_LABEL_DIR, ignore_errors=True)
            ensure_store()
            store.files = []
            store.annotations = []

        for image_path, width, height, detected_content_type in readable_images:
            suffix = image_path.suffix.lower()
            record = FileRecord(
                filename=image_path.name,
                content_type=detected_content_type,
                width=width,
                height=height,
            )
            target = UPLOAD_DIR / f"{record.id}{suffix}"
            shutil.copy2(image_path, target)

            store.files.append(record)
            records.append(record)

        write_store(store)
    return records


@app.get("/api/files/{file_id}/content")
def file_content(file_id: str) -> FileResponse:
    store = read_store()
    record = next((item for item in store.files if item.id == file_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    match = uploaded_path(file_id)
    if match is None:
        raise HTTPException(status_code=404, detail="File content not found")
    return FileResponse(match, media_type=record.content_type, filename=record.filename)


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str) -> dict[str, int | str]:
    store = read_store()
    if not any(item.id == file_id for item in store.files):
        raise HTTPException(status_code=404, detail="File not found")

    deleted_files = 0
    for match in UPLOAD_DIR.glob(f"{file_id}.*"):
        match.unlink()
        deleted_files += 1

    annotation_count = sum(1 for item in store.annotations if item.file_id == file_id)
    store.files = [item for item in store.files if item.id != file_id]
    store.annotations = [item for item in store.annotations if item.file_id != file_id]
    write_store(store)
    return {
        "status": "deleted",
        "files_deleted": deleted_files,
        "annotations_removed": annotation_count,
    }


@app.patch("/api/files/{file_id}")
def update_file(file_id: str, metadata: FileMetadata) -> FileRecord:
    store = read_store()
    record = next((item for item in store.files if item.id == file_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    if record.width == metadata.width and record.height == metadata.height:
        return record

    record.width = metadata.width
    record.height = metadata.height
    write_store(store)
    return record


@app.post("/api/files/{file_id}/resize")
def resize_file(file_id: str, payload: ImageResizePayload) -> FileRecord:
    store = read_store()
    record = next((item for item in store.files if item.id == file_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    source = uploaded_path(file_id)
    if source is None:
        raise HTTPException(status_code=404, detail="File content not found")

    with Image.open(source) as image:
        original_width = record.width or image.width
        original_height = record.height or image.height
        resized = image.resize((payload.width, payload.height), Image.Resampling.LANCZOS)
        if source.suffix.lower() in {".jpg", ".jpeg"} and resized.mode in {"RGBA", "LA", "P"}:
            resized = resized.convert("RGB")
        resized.save(source)

    width_scale = payload.width / original_width
    height_scale = payload.height / original_height
    for annotation in store.annotations:
        if annotation.file_id != file_id:
            continue
        annotation.x = round(annotation.x * width_scale, 2)
        annotation.y = round(annotation.y * height_scale, 2)
        annotation.width = round(annotation.width * width_scale, 2)
        annotation.height = round(annotation.height * height_scale, 2)

    record.width = payload.width
    record.height = payload.height
    write_store(store)
    return record


@app.post("/api/files/{file_id}/augment/rotate")
def augment_rotate_file(file_id: str, payload: RotationAugmentPayload) -> list[FileRecord]:
    allowed_degrees = {15, 45, 90, 180, 270, 315, 345}
    degrees = []
    for degree in payload.degrees:
        if degree not in allowed_degrees:
            raise HTTPException(status_code=400, detail=f"Unsupported rotation degree: {degree}")
        if degree not in degrees:
            degrees.append(degree)

    if not degrees:
        raise HTTPException(status_code=400, detail="Select at least one rotation degree")

    store = read_store()
    record = next((item for item in store.files if item.id == file_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    source = uploaded_path(file_id)
    if source is None:
        raise HTTPException(status_code=404, detail="File content not found")

    file_annotations = [item for item in store.annotations if item.file_id == file_id]
    created_records: list[FileRecord] = []

    with Image.open(source) as image:
        original_width = record.width or image.width
        original_height = record.height or image.height
        suffix = source.suffix.lower()
        stem = Path(record.filename).stem or "image"

        for degree in degrees:
            rotated = image.rotate(degree, expand=True)
            rotated_width = rotated.width
            rotated_height = rotated.height
            output_width = payload.width or rotated_width
            output_height = payload.height or rotated_height

            if (output_width, output_height) != (rotated_width, rotated_height):
                rotated = rotated.resize((output_width, output_height), Image.Resampling.LANCZOS)

            if suffix in {".jpg", ".jpeg"} and rotated.mode in {"RGBA", "LA", "P"}:
                rotated = rotated.convert("RGB")

            rotated_record = FileRecord(
                filename=f"{stem}_rot{degree}{suffix}",
                content_type=record.content_type,
                width=output_width,
                height=output_height,
            )
            target = UPLOAD_DIR / f"{rotated_record.id}{suffix}"
            rotated.save(target)

            for annotation in file_annotations:
                x, y, width, height = rotated_bbox(
                    annotation,
                    original_width,
                    original_height,
                    rotated_width,
                    rotated_height,
                    degree,
                )
                x_scale = output_width / rotated_width
                y_scale = output_height / rotated_height
                store.annotations.append(
                    BoundingBox(
                        file_id=rotated_record.id,
                        label_id=annotation.label_id,
                        type="bbox",
                        x=round(x * x_scale, 2),
                        y=round(y * y_scale, 2),
                        width=round(width * x_scale, 2),
                        height=round(height * y_scale, 2),
                    )
                )

            store.files.append(rotated_record)
            created_records.append(rotated_record)

    write_store(store)
    write_yolo_source_folder(store)
    return created_records


@app.post("/api/ml/yolo/train")
def start_yolo_training(payload: YoloTrainPayload) -> dict[str, Any]:
    with TRAINING_LOCK:
        if TRAINING_STATUS["running"]:
            raise HTTPException(status_code=409, detail="YOLO training is already running")
        TRAINING_STATUS.update(
            {
                "running": True,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": None,
                "message": "Training queued.",
                "error": None,
            }
        )

    thread = threading.Thread(target=run_yolo_training, args=(payload,), daemon=True)
    thread.start()
    return {"status": "started", **TRAINING_STATUS}


@app.get("/api/ml/yolo/status")
def yolo_training_status() -> dict[str, Any]:
    with TRAINING_LOCK:
        status = dict(TRAINING_STATUS)
    status["model_exists"] = BEST_MODEL_PATH.exists()
    if status.get("model_path") is None and BEST_MODEL_PATH.exists():
        status["model_path"] = str(BEST_MODEL_PATH)
    return status


@app.get("/api/ml/yolo/environment")
def yolo_environment() -> dict[str, Any]:
    return gpu_environment()


@app.post("/api/ml/yolo/auto-annotate")
def auto_annotate_with_yolo(payload: AutoAnnotatePayload) -> dict[str, Any]:
    if not BEST_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="Train a YOLO model before auto annotation")

    YOLO = load_yolo_class()
    model = YOLO(str(BEST_MODEL_PATH))
    try:
        device = resolve_yolo_device(payload.device)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with STORE_LOCK:
        store = read_store()
        if not store.labels:
            raise HTTPException(status_code=400, detail="No labels are available")

        selected_ids = set(payload.file_ids or [file.id for file in store.files])
        files_by_id = {file.id: file for file in store.files}
        unknown_ids = selected_ids - set(files_by_id)
        if unknown_ids:
            raise HTTPException(status_code=404, detail=f"Unknown file id: {sorted(unknown_ids)[0]}")

        new_annotations: list[BoundingBox] = []
        files_processed = 0
        for file_id in selected_ids:
            file = files_by_id[file_id]
            source = uploaded_path(file.id)
            if source is None:
                continue
            predicted_annotations = predicted_boxes_for_file(
                model,
                file,
                source,
                store.labels,
                payload.confidence,
                device,
            )
            if payload.replace_existing:
                store.annotations = [annotation for annotation in store.annotations if annotation.file_id != file.id]
            store.annotations.extend(predicted_annotations)
            new_annotations.extend(predicted_annotations)
            files_processed += 1

        write_store(store)

    return {
        "files_processed": files_processed,
        "annotations_added": len(new_annotations),
        "replace_existing": payload.replace_existing,
        "confidence": payload.confidence,
        "device": device,
    }


@app.put("/api/files/{file_id}/annotations")
def save_annotations(file_id: str, payload: AnnotationPayload) -> dict[str, Any]:
    store = read_store()
    record = next((item for item in store.files if item.id == file_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    label_ids = {label.id for label in store.labels}
    valid_annotations: list[BoundingBox] = []
    skipped_annotations = 0
    image_width = float(record.width or 0)
    image_height = float(record.height or 0)

    for raw_annotation in payload.annotations:
        annotation_data = {**raw_annotation, "file_id": file_id, "type": "bbox"}
        try:
            annotation = BoundingBox.model_validate(annotation_data)
        except ValueError:
            skipped_annotations += 1
            continue

        if annotation.file_id != file_id:
            skipped_annotations += 1
            continue
        if annotation.label_id not in label_ids:
            skipped_annotations += 1
            continue
        if not all(math.isfinite(value) for value in (annotation.x, annotation.y, annotation.width, annotation.height)):
            skipped_annotations += 1
            continue
        if annotation.width <= 0 or annotation.height <= 0:
            skipped_annotations += 1
            continue

        left = max(0.0, annotation.x)
        top = max(0.0, annotation.y)
        right = annotation.x + annotation.width
        bottom = annotation.y + annotation.height
        if image_width > 0:
            right = min(image_width, right)
        if image_height > 0:
            bottom = min(image_height, bottom)
        clipped_width = round(right - left, 2)
        clipped_height = round(bottom - top, 2)
        if clipped_width <= 0 or clipped_height <= 0:
            skipped_annotations += 1
            continue

        annotation.x = round(left, 2)
        annotation.y = round(top, 2)
        annotation.width = clipped_width
        annotation.height = clipped_height
        valid_annotations.append(annotation)

    if payload.annotations and not valid_annotations:
        raise HTTPException(status_code=400, detail="No valid annotations to save")

    store.annotations = [item for item in store.annotations if item.file_id != file_id]
    store.annotations.extend(valid_annotations)
    write_store(store)
    return {"annotations": valid_annotations, "skipped_annotations": skipped_annotations}


@app.get("/api/export/coco")
def export_coco() -> dict[str, Any]:
    store = read_store()
    labels_by_id = {label.id: label for label in store.labels}
    files_by_id = {file.id: file for file in store.files}

    categories = [{"id": index + 1, "name": label.name} for index, label in enumerate(store.labels)]
    category_ids = {label.id: index + 1 for index, label in enumerate(store.labels)}
    images = [
        {"id": index + 1, "file_name": file.filename, "width": file.width or 0, "height": file.height or 0}
        for index, file in enumerate(store.files)
    ]
    image_ids = {file.id: index + 1 for index, file in enumerate(store.files)}

    annotations = []
    for index, annotation in enumerate(store.annotations):
        if annotation.label_id not in labels_by_id or annotation.file_id not in files_by_id:
            continue
        annotations.append(
            {
                "id": index + 1,
                "image_id": image_ids[annotation.file_id],
                "category_id": category_ids[annotation.label_id],
                "bbox": [annotation.x, annotation.y, annotation.width, annotation.height],
                "area": annotation.width * annotation.height,
                "iscrowd": 0,
            }
        )

    return {"images": images, "annotations": annotations, "categories": categories}


@app.get("/api/export/yolo")
def export_yolo() -> dict[str, str]:
    return build_yolo_label_map(read_store())


@app.get("/api/export/yolo-seg")
def export_yolo_seg() -> dict[str, str]:
    return build_yolo_seg_label_map(read_store())


@app.get("/api/export/yolo.zip")
def export_yolo_zip() -> StreamingResponse:
    store = read_store()
    headers = {"Content-Disposition": 'attachment; filename="yolo-dataset.zip"'}
    return StreamingResponse(io.BytesIO(build_yolo_zip(store)), media_type="application/zip", headers=headers)


@app.get("/api/export/yolo-seg.zip")
def export_yolo_seg_zip() -> StreamingResponse:
    store = read_store()
    headers = {"Content-Disposition": 'attachment; filename="yolo-seg-dataset.zip"'}
    return StreamingResponse(io.BytesIO(build_yolo_seg_zip(store)), media_type="application/zip", headers=headers)


@app.post("/api/export/yolo/source-folder")
def export_yolo_source_folder() -> dict[str, int | str]:
    return write_yolo_source_folder(read_store())


@app.post("/api/export/yolo-seg/source-folder")
def export_yolo_seg_source_folder() -> dict[str, int | str]:
    return write_yolo_seg_source_folder(read_store())
