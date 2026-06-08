# Annotation MVP

A compact image annotation prototype with manual bounding boxes, class labels, saved annotations, multi-image box copy, and COCO/YOLO exports.

## Run the backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Run the Python UI

```powershell
cd frontend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

Use Python 3.11 for both environments. On this machine, the default `python`
points to Python 3.14, which is too new for this pinned Streamlit/FastAPI stack.

## Annotation workflow

1. Start the backend and Streamlit UI.
2. Upload images or import an image folder from the sidebar.
3. Create one or more class labels.
4. Select an image, choose the active label, and draw boxes on the canvas.
5. Click `Save annotations` or `Save and next`.

The app saves valid boxes even if one canvas object is incomplete or invalid. Invalid boxes are skipped and the UI shows a warning with the skipped count.

Use `Apply boxes to selected images` to copy the current image's boxes to other images. Enable `Apply to all other images` to copy them across the rest of the dataset. Copied boxes are clipped to each target image size.

## MVP Scope

- Upload image files.
- Create multiclass labels.
- Draw bounding boxes on the selected image.
- Save complete or partial valid annotations through the FastAPI backend.
- Copy current boxes to selected images or all other images.
- Export COCO JSON and YOLO-style label content.
- Save YOLO or YOLO-Seg source folders.

The backend currently uses local JSON storage in `backend/data/db.json` and uploaded files in `backend/data/uploads`. The API shape is designed so PostgreSQL and object storage can replace those pieces later.

## Export options

The sidebar supports:

- COCO JSON download.
- YOLO labels JSON download.
- YOLO-Seg labels JSON download.
- YOLO dataset ZIP download.
- YOLO-Seg dataset ZIP download.
- Writing YOLO or YOLO-Seg files to `backend/data/source`.
