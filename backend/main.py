from __future__ import annotations

import io
import json
import math
import os
import shutil
import stat
import threading
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
DB_PATH = DATA_DIR / "db.json"
STORE_LOCK = threading.Lock()

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
    (SOURCE_DIR / "data.yaml").write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(store.labels)}\n"
        f"names: {[label.name for label in store.labels]!r}\n",
        encoding="utf-8",
    )

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
    (SOURCE_DIR / "data.yaml").write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(store.labels)}\n"
        f"names: {[label.name for label in store.labels]!r}\n",
        encoding="utf-8",
    )

    return {
        "folder": str(SOURCE_DIR),
        "image_folder": str(SOURCE_IMAGE_DIR),
        "label_folder": str(SOURCE_LABEL_DIR),
        "images_written": image_count,
        "labels_written": label_count,
    }


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
