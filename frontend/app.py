from __future__ import annotations

import io
import json
import hashlib
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from streamlit.elements.lib.image_utils import image_to_url
import streamlit.elements.image as st_image
from PIL import Image
from streamlit_drawable_canvas import st_canvas


API_URL = os.environ.get("ANNOTATION_API_URL", "http://127.0.0.1:8000")
COLORS = [
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
ROTATION_DEGREES = [15, 45, 90, 180, 270, 315, 345]


st.set_page_config(page_title="Annotation MVP", layout="wide")


def install_canvas_image_compat() -> None:
    """Restore the image_to_url helper expected by streamlit-drawable-canvas."""
    if hasattr(st_image, "image_to_url"):
        return

    st_image.image_to_url = image_to_url


install_canvas_image_compat()


def api(method: str, path: str, **kwargs: Any) -> requests.Response:
    request_timeout = kwargs.pop("request_timeout", 20)
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=request_timeout, **kwargs)
    except requests.Timeout as exc:
        st.error(f"API request timed out after {request_timeout} seconds. Try a smaller folder or wait for the backend to finish. Details: {exc}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Backend is not reachable at {API_URL}. Start FastAPI first. Details: {exc}")
        st.stop()

    if not response.ok:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"API error: {detail}")
        st.stop()
    return response


def load_state() -> dict[str, Any]:
    return api("GET", "/api/state").json()


def upload_image(uploaded_file: Any) -> None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    api("POST", "/api/files", files=files, request_timeout=120)


def import_folder(folder_path: str, recursive: bool, replace_existing: bool) -> list[dict[str, Any]]:
    return api(
        "POST",
        "/api/folders/import",
        request_timeout=600,
        json={
            "folder_path": folder_path,
            "recursive": recursive,
            "replace_existing": replace_existing,
        },
    ).json()


def reset_dataset_state() -> None:
    for key in list(st.session_state):
        if (
            key.startswith("canvas-")
            or key.startswith("resize-")
            or key.startswith("rotation-")
            or key.startswith("label-")
            or key.startswith("x-")
            or key.startswith("y-")
            or key.startswith("width-")
            or key.startswith("height-")
            or key
            in {
                "selected_file_id",
                "selected_file_index",
                "coco_export",
                "yolo_export",
                "yolo_seg_export",
                "yolo_zip_export",
                "yolo_seg_zip_export",
            }
        ):
            del st.session_state[key]


def color_for_label(existing_labels: list[dict[str, Any]]) -> str:
    used_colors = {label["color"].lower() for label in existing_labels}
    for color in COLORS:
        if color.lower() not in used_colors:
            return color
    return COLORS[len(existing_labels) % len(COLORS)]


def add_label(name: str, existing_labels: list[dict[str, Any]]) -> None:
    payload = {"name": name.strip(), "color": color_for_label(existing_labels)}
    api("POST", "/api/labels", json=payload)


def update_label(label_id: str, name: str, color: str) -> None:
    api("PATCH", f"/api/labels/{label_id}", json={"name": name.strip(), "color": color})


def delete_label(label_id: str) -> None:
    api("DELETE", f"/api/labels/{label_id}")


def delete_file(file_id: str) -> None:
    api("DELETE", f"/api/files/{file_id}")


def image_bytes(file_id: str) -> bytes:
    return api("GET", f"/api/files/{file_id}/content").content


def sync_image_size(file_id: str, image: Image.Image) -> None:
    api("PATCH", f"/api/files/{file_id}", json={"width": image.width, "height": image.height})


def resize_image(file_id: str, width: int, height: int) -> None:
    api("POST", f"/api/files/{file_id}/resize", json={"width": width, "height": height})


def create_rotations(file_id: str, degrees: list[int]) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"degrees": degrees}
    target = fixed_resize_target()
    if target is not None:
        payload["width"] = target[0]
        payload["height"] = target[1]
    return api("POST", f"/api/files/{file_id}/augment/rotate", json=payload).json()


def scaled_canvas_size(image: Image.Image, max_width: int = 980, max_height: int = 620) -> tuple[int, int, float]:
    scale = min(1.0, max_width / image.width, max_height / image.height)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return width, height, scale


def canvas_initial_drawing(
    annotations: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    scale: float,
) -> dict[str, Any]:
    objects = []
    for annotation in annotations:
        label = labels_by_id.get(annotation["label_id"], {})
        color = label.get("color", "#2563eb")
        objects.append(
            {
                "type": "rect",
                "id": annotation.get("id"),
                "label_id": annotation["label_id"],
                "left": annotation["x"] * scale,
                "top": annotation["y"] * scale,
                "width": annotation["width"] * scale,
                "height": annotation["height"] * scale,
                "fill": "rgba(37, 99, 235, 0.08)",
                "stroke": color,
                "strokeWidth": 3,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def annotations_from_canvas(
    canvas_json: dict[str, Any] | None,
    file_id: str,
    selected_label_id: str,
    labels: list[dict[str, Any]],
    existing_annotations: list[dict[str, Any]],
    scale: float,
) -> list[dict[str, Any]]:
    if not canvas_json:
        return existing_annotations

    label_ids = {label["id"] for label in labels}
    labels_by_color = {label["color"].lower(): label["id"] for label in labels}
    annotations = []
    rect_index = 0
    for item in canvas_json.get("objects", []):
        if item.get("type") != "rect":
            continue

        label_id = item.get("label_id")
        annotation_id = item.get("id")
        if label_id not in label_ids and rect_index < len(existing_annotations):
            label_id = existing_annotations[rect_index]["label_id"]
        if label_id not in label_ids:
            stroke = str(item.get("stroke", "")).lower()
            label_id = labels_by_color.get(stroke)
        if label_id not in label_ids:
            label_id = selected_label_id
        try:
            scale_x = float(item.get("scaleX", 1))
            scale_y = float(item.get("scaleY", 1))
            canvas_left = float(item.get("left", 0))
            canvas_top = float(item.get("top", 0))
            canvas_width = float(item.get("width", 0)) * scale_x
            canvas_height = float(item.get("height", 0)) * scale_y
        except (TypeError, ValueError):
            rect_index += 1
            continue
        left = min(canvas_left, canvas_left + canvas_width) / scale
        top = min(canvas_top, canvas_top + canvas_height) / scale
        width = abs(canvas_width) / scale
        height = abs(canvas_height) / scale
        if width < 1 or height < 1:
            continue

        annotations.append(
            {
                **({"id": annotation_id} if annotation_id else {}),
                "file_id": file_id,
                "label_id": label_id,
                "type": "bbox",
                "x": round(left),
                "y": round(top),
                "width": round(width),
                "height": round(height),
            }
        )
        rect_index += 1
    return annotations


def save_annotations(file_id: str, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = api("PUT", f"/api/files/{file_id}/annotations", json={"annotations": annotations}).json()
    if isinstance(result, list):
        return result

    skipped_annotations = int(result.get("skipped_annotations", 0) or 0)
    if skipped_annotations:
        st.warning(f"Skipped {skipped_annotations} invalid or incomplete box(es). Saved the valid boxes.")
    return result.get("annotations", [])


def clamp_annotations_to_image(
    annotations: list[dict[str, Any]],
    file_id: str,
    width: int,
    height: int,
    *,
    keep_ids: bool = True,
) -> list[dict[str, Any]]:
    clipped_annotations = []
    for annotation in annotations:
        left = max(0.0, float(annotation["x"]))
        top = max(0.0, float(annotation["y"]))
        right = min(float(width), float(annotation["x"]) + float(annotation["width"]))
        bottom = min(float(height), float(annotation["y"]) + float(annotation["height"]))
        clipped_width = round(right - left)
        clipped_height = round(bottom - top)
        if clipped_width < 1 or clipped_height < 1:
            continue

        clipped_annotation = {
            "file_id": file_id,
            "label_id": annotation["label_id"],
            "type": "bbox",
            "x": round(left),
            "y": round(top),
            "width": clipped_width,
            "height": clipped_height,
        }
        if keep_ids and annotation.get("id"):
            clipped_annotation["id"] = annotation["id"]
        clipped_annotations.append(clipped_annotation)
    return clipped_annotations


def fixed_resize_target() -> tuple[int, int] | None:
    if not st.session_state.get("fixed_resize_enabled"):
        return None

    width = int(st.session_state.get("fixed_resize_width", 0) or 0)
    height = int(st.session_state.get("fixed_resize_height", 0) or 0)
    if width < 1 or height < 1:
        return None
    return width, height


def save_annotations_and_apply_fixed_size(
    file_id: str,
    annotations: list[dict[str, Any]],
    image: Image.Image,
) -> bool:
    save_annotations(file_id, clamp_annotations_to_image(annotations, file_id, image.width, image.height))
    target = fixed_resize_target()
    if target is None or target == (image.width, image.height):
        return False

    resize_image(file_id, target[0], target[1])
    return True


def save_annotations_for_files(
    file_ids: list[str],
    annotations: list[dict[str, Any]],
    files_by_id: dict[str, dict[str, Any]],
) -> None:
    for file_id in file_ids:
        target_file = files_by_id[file_id]
        copied_annotations = clamp_annotations_to_image(
            annotations,
            file_id,
            int(target_file.get("width") or 1),
            int(target_file.get("height") or 1),
            keep_ids=False,
        )
        save_annotations(file_id, copied_annotations)


def replace_annotation(
    annotations: list[dict[str, Any]],
    index: int,
    updated_annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    next_annotations = annotations.copy()
    next_annotations[index] = updated_annotation
    return next_annotations


def delete_annotation(annotations: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    return [annotation for item_index, annotation in enumerate(annotations) if item_index != index]


def refresh_canvas(file_id: str) -> None:
    key = f"canvas-version-{file_id}"
    st.session_state[key] = st.session_state.get(key, 0) + 1


def export_payload(format_name: str) -> str:
    return json.dumps(api("GET", f"/api/export/{format_name}").json(), indent=2)


def export_file(path: str) -> bytes:
    return api("GET", path, request_timeout=120).content


def save_yolo_source_folder() -> dict[str, Any]:
    return api("POST", "/api/export/yolo/source-folder").json()


def save_yolo_seg_source_folder() -> dict[str, Any]:
    return api("POST", "/api/export/yolo-seg/source-folder").json()


st.title("Annotation MVP")
st.caption("Python UI for image upload, class labels, bounding boxes, save, and export.")

state = load_state()
labels = state["labels"]
files = state["files"]
annotations = state["annotations"]
labels_by_id = {label["id"]: label for label in labels}
files_by_id = {file["id"]: file for file in files}

with st.sidebar:
    st.header("Dataset")
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp", "bmp"], accept_multiple_files=True)
    if st.button("Upload selected images", disabled=not uploaded_files, use_container_width=True):
        reset_dataset_state()
        for uploaded_file in uploaded_files:
            upload_image(uploaded_file)
        st.session_state.selected_file_index = 0
        st.session_state.selected_file_id = None
        st.rerun()

    with st.expander("Import image folder", expanded=False):
        folder_path = st.text_input("Folder path")
        recursive_import = st.checkbox("Include subfolders")
        replace_import = st.checkbox("Replace previous images", value=True)
        if st.button("Import folder", disabled=not folder_path.strip(), use_container_width=True):
            reset_dataset_state()
            imported_files = import_folder(folder_path.strip(), recursive_import, replace_import)
            st.session_state.selected_file_index = 0
            st.session_state.selected_file_id = imported_files[0]["id"] if imported_files else None
            st.session_state.current_folder_path = folder_path.strip()
            st.success(f"Imported {len(imported_files)} image(s).")
            st.rerun()

    st.divider()
    st.header("Labels")
    with st.form("label-form", clear_on_submit=True):
        label_name = st.text_input("New class label")
        submitted = st.form_submit_button("Add label", use_container_width=True)
        if submitted and label_name.strip():
            add_label(label_name, labels)
            st.rerun()

    if labels:
        selected_label_name = st.radio("Active label", [label["name"] for label in labels])
        selected_label = next(label for label in labels if label["name"] == selected_label_name)
        st.color_picker("Box color", selected_label["color"], disabled=True)

        with st.expander("Edit or delete class", expanded=False):
            editable_label_name = st.selectbox(
                "Class",
                [label["name"] for label in labels],
                index=[label["id"] for label in labels].index(selected_label["id"]),
                key="editable-label",
            )
            editable_label = next(label for label in labels if label["name"] == editable_label_name)
            edited_label_name = st.text_input("Class name", editable_label["name"], key=f"edit-name-{editable_label['id']}")
            edited_label_color = st.color_picker(
                "Class color",
                editable_label["color"],
                key=f"edit-color-{editable_label['id']}",
            )
            used_box_count = sum(1 for annotation in annotations if annotation["label_id"] == editable_label["id"])

            update_class_col, delete_class_col = st.columns(2)
            with update_class_col:
                update_disabled = not edited_label_name.strip()
                if st.button("Update class", disabled=update_disabled, use_container_width=True):
                    update_label(editable_label["id"], edited_label_name, edited_label_color)
                    st.success("Class updated.")
                    st.rerun()

            with delete_class_col:
                confirm_delete = st.checkbox(
                    f"Delete {used_box_count} box(es)",
                    key=f"delete-label-confirm-{editable_label['id']}",
                )
                if st.button(
                    "Delete class",
                    disabled=not confirm_delete,
                    use_container_width=True,
                ):
                    delete_label(editable_label["id"])
                    reset_dataset_state()
                    st.success("Class deleted.")
                    st.rerun()
    else:
        selected_label = None
        st.info("Create a label before drawing boxes.")

    st.divider()
    st.header("Exports")
    if st.button("Prepare COCO JSON", use_container_width=True):
        st.session_state.coco_export = export_payload("coco")
    if "coco_export" in st.session_state:
        st.download_button(
            "Download COCO JSON",
            st.session_state.coco_export,
            "annotations-coco.json",
            "application/json",
        )

    if st.button("Prepare YOLO labels JSON", use_container_width=True):
        st.session_state.yolo_export = export_payload("yolo")
    if "yolo_export" in st.session_state:
        st.download_button(
            "Download YOLO labels JSON",
            st.session_state.yolo_export,
            "annotations-yolo.json",
            "application/json",
        )

    if st.button("Prepare YOLO-Seg labels JSON", use_container_width=True):
        st.session_state.yolo_seg_export = export_payload("yolo-seg")
    if "yolo_seg_export" in st.session_state:
        st.download_button(
            "Download YOLO-Seg labels JSON",
            st.session_state.yolo_seg_export,
            "annotations-yolo-seg.json",
            "application/json",
        )

    if st.button("Prepare YOLO dataset ZIP", use_container_width=True):
        st.session_state.yolo_zip_export = export_file("/api/export/yolo.zip")
    if "yolo_zip_export" in st.session_state:
        st.download_button(
            "Download YOLO dataset ZIP",
            st.session_state.yolo_zip_export,
            "yolo-dataset.zip",
            "application/zip",
        )
    if st.button("Prepare YOLO-Seg dataset ZIP", use_container_width=True):
        st.session_state.yolo_seg_zip_export = export_file("/api/export/yolo-seg.zip")
    if "yolo_seg_zip_export" in st.session_state:
        st.download_button(
            "Download YOLO-Seg dataset ZIP",
            st.session_state.yolo_seg_zip_export,
            "yolo-seg-dataset.zip",
            "application/zip",
        )
    if st.button("Save YOLO files to source folder", use_container_width=True):
        result = save_yolo_source_folder()
        st.success(
            f"Saved {result['images_written']} image(s) to {result['image_folder']} and "
            f"{result['labels_written']} label file(s) to {result['label_folder']}."
        )
    if st.button("Save YOLO-Seg files to source folder", use_container_width=True):
        result = save_yolo_seg_source_folder()
        st.success(
            f"Saved {result['images_written']} image(s) to {result['image_folder']} and "
            f"{result['labels_written']} label file(s) to {result['label_folder']}."
        )

if not files:
    st.info("Upload an image from the sidebar to begin.")
    st.stop()

dataset_signature = hashlib.sha1("|".join(file["id"] for file in files).encode("utf-8")).hexdigest()
file_ids = [file["id"] for file in files]
if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state.dataset_signature = dataset_signature
    if st.session_state.get("selected_file_id") in file_ids:
        st.session_state.selected_file_index = file_ids.index(st.session_state.selected_file_id)
    else:
        st.session_state.selected_file_index = 0
        st.session_state.selected_file_id = files[0]["id"]

if st.session_state.get("selected_file_id") not in file_ids:
    st.session_state.selected_file_id = files[0]["id"]

selected_file_index = file_ids.index(st.session_state.selected_file_id)
st.session_state.selected_file_index = selected_file_index
try:
    selected_file_index = int(st.session_state.selected_file_index)
except (TypeError, ValueError):
    selected_file_index = 0
selected_file_index = max(0, min(selected_file_index, len(files) - 1))
st.session_state.selected_file_index = selected_file_index
st.session_state.selected_file_id = files[selected_file_index]["id"]

nav_prev_col, image_select_col, nav_next_col = st.columns([1, 5, 1])
with nav_prev_col:
    if st.button("Previous", disabled=st.session_state.selected_file_index <= 0, use_container_width=True):
        st.session_state.selected_file_index -= 1
        st.session_state.selected_file_id = files[st.session_state.selected_file_index]["id"]
        st.rerun()

file_options = [f"{index + 1}. {file['filename']}" for index, file in enumerate(files)]
with image_select_col:
    selected_option = st.selectbox(
        "Image",
        file_options,
        index=selected_file_index,
        key=f"image-select-{dataset_signature}",
    )
    st.session_state.selected_file_index = file_options.index(selected_option)
    st.session_state.selected_file_id = files[st.session_state.selected_file_index]["id"]

with nav_next_col:
    if st.button("Next", disabled=st.session_state.selected_file_index >= len(files) - 1, use_container_width=True):
        st.session_state.selected_file_index += 1
        st.session_state.selected_file_id = files[st.session_state.selected_file_index]["id"]
        st.rerun()

selected_file = files[st.session_state.selected_file_index]
selected_file_id = selected_file["id"]
file_annotations = [item for item in annotations if item["file_id"] == selected_file_id]

with st.expander("Selected image actions", expanded=False):
    delete_image_confirm = st.checkbox("Delete this image and its boxes", key=f"delete-file-confirm-{selected_file_id}")
    if st.button("Delete selected image", disabled=not delete_image_confirm, use_container_width=True):
        delete_file(selected_file_id)
        reset_dataset_state()
        st.success("Image deleted.")
        st.rerun()

raw_image = image_bytes(selected_file_id)
image = Image.open(io.BytesIO(raw_image)).convert("RGB")
sync_image_size(selected_file_id, image)

left, right = st.columns([3, 1], gap="large")

with left:
    st.subheader("Editor")
    fit_col, zoom_col = st.columns([1, 1])
    with fit_col:
        fit_to_screen = st.checkbox("Fit image to screen", value=True)
    with zoom_col:
        display_zoom = st.slider(
            "Display zoom",
            min_value=25,
            max_value=150,
            value=100,
            step=5,
            disabled=fit_to_screen,
        )

    with st.expander("Resize image", expanded=False):
        fixed_size_enabled = st.checkbox(
            "Fix size for saves",
            value=bool(st.session_state.get("fixed_resize_enabled", False)),
            key="fixed_resize_enabled",
        )
        keep_aspect = st.checkbox("Keep aspect ratio", value=True, disabled=fixed_size_enabled)
        if fixed_size_enabled:
            keep_aspect = False
        if fixed_size_enabled and "fixed_resize_width" not in st.session_state:
            st.session_state.fixed_resize_width = image.width
        if fixed_size_enabled and "fixed_resize_height" not in st.session_state:
            st.session_state.fixed_resize_height = image.height

        width_value = (
            int(st.session_state.get("fixed_resize_width", image.width))
            if fixed_size_enabled
            else image.width
        )
        resize_width = st.number_input(
            "Width",
            min_value=1,
            max_value=10000,
            value=width_value,
            step=1,
            key="resize-width-fixed" if fixed_size_enabled else f"resize-width-{selected_file_id}",
        )
        if keep_aspect:
            resize_height = max(1, round(resize_width * image.height / image.width))
            st.number_input(
                "Height",
                min_value=1,
                max_value=10000,
                value=resize_height,
                step=1,
                disabled=True,
                key="resize-height-locked-fixed" if fixed_size_enabled else f"resize-height-locked-{selected_file_id}",
            )
        else:
            height_value = (
                int(st.session_state.get("fixed_resize_height", image.height))
                if fixed_size_enabled
                else image.height
            )
            resize_height = st.number_input(
                "Height",
                min_value=1,
                max_value=10000,
                value=height_value,
                step=1,
                key="resize-height-fixed" if fixed_size_enabled else f"resize-height-{selected_file_id}",
            )

        if fixed_size_enabled:
            st.session_state.fixed_resize_width = int(resize_width)
            st.session_state.fixed_resize_height = int(resize_height)

        resize_disabled = resize_width == image.width and resize_height == image.height
        if st.button("Apply resize", disabled=resize_disabled, use_container_width=True):
            resize_image(selected_file_id, int(resize_width), int(resize_height))
            st.success("Image resized and boxes scaled.")
            st.rerun()

        if fixed_size_enabled:
            st.caption(f"Saving annotations will resize this image to {int(resize_width)} x {int(resize_height)}.")

    if fit_to_screen:
        canvas_width, canvas_height, scale = scaled_canvas_size(image)
    else:
        zoom_scale = display_zoom / 100
        canvas_width = max(1, int(image.width * zoom_scale))
        canvas_height = max(1, int(image.height * zoom_scale))
        scale = zoom_scale

    background = image.resize((canvas_width, canvas_height))
    canvas_version = st.session_state.get(f"canvas-version-{selected_file_id}", 0)
    canvas_key = f"canvas-{selected_file_id}-{canvas_width}x{canvas_height}-{canvas_version}"

    if selected_label is None:
        st.warning("Create and select a label to enable box drawing.")
        drawing_mode = "transform"
        stroke_color = "#2563eb"
    else:
        drawing_mode = "rect"
        stroke_color = selected_label["color"]

    canvas_result = st_canvas(
        fill_color="rgba(37, 99, 235, 0.08)",
        stroke_width=3,
        stroke_color=stroke_color,
        background_image=background,
        initial_drawing=canvas_initial_drawing(file_annotations, labels_by_id, scale),
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        key=canvas_key,
    )

    current_annotations = annotations_from_canvas(
        canvas_result.json_data,
        selected_file_id,
        selected_label["id"] if selected_label else "",
        labels,
        file_annotations,
        scale,
    )
    save_col, save_next_col = st.columns(2)
    with save_col:
        if st.button("Save annotations", type="primary", disabled=selected_label is None, use_container_width=True):
            resized_on_save = save_annotations_and_apply_fixed_size(selected_file_id, current_annotations, image)
            if resized_on_save:
                target_width, target_height = fixed_resize_target() or (image.width, image.height)
                st.success(f"Annotations saved and image resized to {target_width} x {target_height}.")
            else:
                st.success("Annotations saved.")
            st.rerun()
    with save_next_col:
        if st.button(
            "Save and next",
            disabled=selected_label is None or st.session_state.selected_file_index >= len(files) - 1,
            use_container_width=True,
        ):
            save_annotations_and_apply_fixed_size(selected_file_id, current_annotations, image)
            st.session_state.selected_file_index += 1
            st.session_state.selected_file_id = files[st.session_state.selected_file_index]["id"]
            st.rerun()

    with st.expander("Create rotated copies", expanded=False):
        st.caption("Creates new image entries and rotates all boxes for this image.")
        rotation_cols = st.columns(len(ROTATION_DEGREES))
        selected_degrees = []
        for degree, column in zip(ROTATION_DEGREES, rotation_cols):
            with column:
                if st.checkbox(f"{degree}", key=f"rotation-{selected_file_id}-{degree}"):
                    selected_degrees.append(degree)

        if st.button(
            "Generate rotations",
            disabled=not selected_degrees or not current_annotations,
            use_container_width=True,
        ):
            save_annotations_and_apply_fixed_size(selected_file_id, current_annotations, image)
            created_files = create_rotations(selected_file_id, selected_degrees)
            st.session_state.selected_file_id = selected_file_id
            st.session_state.selected_file_index = file_ids.index(selected_file_id)
            st.success(f"Created {len(created_files)} rotated image(s).")
            st.rerun()

    st.divider()
    st.subheader("Apply to multiple images")
    target_files = [file for file in files if file["id"] != selected_file_id]
    target_names = [file["filename"] for file in target_files]
    apply_to_all_targets = st.checkbox("Apply to all other images", disabled=not target_files)
    selected_targets = st.multiselect(
        "Copy current boxes to",
        target_names,
        disabled=apply_to_all_targets,
    )
    if apply_to_all_targets:
        target_ids = [file["id"] for file in target_files]
    else:
        target_ids = [file["id"] for file in target_files if file["filename"] in selected_targets]
    if st.button(
        "Apply boxes to selected images",
        disabled=not current_annotations or not target_ids,
        use_container_width=True,
    ):
        save_annotations_for_files(target_ids, current_annotations, files_by_id)
        st.success(f"Copied boxes to {len(target_ids)} image(s).")
        st.rerun()

with right:
    st.subheader("Boxes")
    if not current_annotations:
        st.caption("No boxes on this image yet.")
    for display_index, annotation in enumerate(current_annotations, start=1):
        label = labels_by_id.get(annotation["label_id"], {"name": "Unknown", "color": "#64748b"})
        annotation_index = display_index - 1
        with st.expander(
            f"#{display_index} {label['name']} - {annotation['x']}, {annotation['y']}, {annotation['width']} x {annotation['height']}",
            expanded=False,
        ):
            if labels:
                label_names = [item["name"] for item in labels]
                current_label_index = next(
                    (item_index for item_index, item in enumerate(labels) if item["id"] == annotation["label_id"]),
                    0,
                )
                edited_label_name = st.selectbox(
                    "Label",
                    label_names,
                    index=current_label_index,
                    key=f"label-{selected_file_id}-{annotation_index}",
                )
                edited_label = next(item for item in labels if item["name"] == edited_label_name)
            else:
                edited_label = label

            edited_x = st.number_input(
                "X",
                min_value=0,
                max_value=max(image.width, 1),
                value=max(0, int(annotation["x"])),
                step=1,
                key=f"x-{selected_file_id}-{annotation_index}",
            )
            edited_y = st.number_input(
                "Y",
                min_value=0,
                max_value=max(image.height, 1),
                value=max(0, int(annotation["y"])),
                step=1,
                key=f"y-{selected_file_id}-{annotation_index}",
            )
            edited_width = st.number_input(
                "Width",
                min_value=1,
                max_value=max(image.width, 1),
                value=max(1, int(annotation["width"])),
                step=1,
                key=f"width-{selected_file_id}-{annotation_index}",
            )
            edited_height = st.number_input(
                "Height",
                min_value=1,
                max_value=max(image.height, 1),
                value=max(1, int(annotation["height"])),
                step=1,
                key=f"height-{selected_file_id}-{annotation_index}",
            )

            update_col, delete_col = st.columns(2)
            with update_col:
                if st.button("Update", key=f"update-{selected_file_id}-{annotation_index}", use_container_width=True):
                    updated_annotation = {
                        **annotation,
                        "label_id": edited_label["id"],
                        "x": edited_x,
                        "y": edited_y,
                        "width": edited_width,
                        "height": edited_height,
                    }
                    save_annotations(
                        selected_file_id,
                        replace_annotation(current_annotations, annotation_index, updated_annotation),
                    )
                    refresh_canvas(selected_file_id)
                    st.success("Box updated.")
                    st.rerun()

            with delete_col:
                if st.button("Delete", key=f"delete-{selected_file_id}-{annotation_index}", use_container_width=True):
                    save_annotations(selected_file_id, delete_annotation(current_annotations, annotation_index))
                    refresh_canvas(selected_file_id)
                    st.success("Box deleted.")
                    st.rerun()

    st.caption(f"Original image: {image.width} x {image.height}")
