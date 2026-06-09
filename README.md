# Annotation MVP

A compact image annotation prototype with manual bounding boxes, class labels, saved annotations, multi-image box copy, YOLO training, auto annotation, and COCO/YOLO exports.

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

## YOLO training and auto annotation

The backend uses Ultralytics YOLO for model training and prediction.

1. Manually annotate a set of images and save the boxes.
2. In the sidebar, open `Auto Annotation`.
3. Open `Train YOLO`, choose a base model, epochs, image size, and batch size.
4. Click `Start training`.
5. When the status says the YOLO model is ready, open `Run auto annotation`.
6. Choose confidence, whether to replace existing boxes, and selected/all images.
7. Click `Auto annotate`.

The app builds an 80/20 train/validation dataset from annotated images under `backend/data/training`. The trained model is saved locally at `backend/data/models/best.pt`, and training artifacts are written under `backend/data/runs`. These folders are ignored by git.

Training can be slow on CPU. Use a CUDA-enabled PyTorch environment if you want practical training speed on larger datasets.

### NVIDIA GPU setup on Windows

Install the normal backend requirements first, then replace CPU-only PyTorch with the CUDA 12.6 build:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --no-cache-dir --force-reinstall --no-deps -r requirements-gpu.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

For an RTX 2050 with 4 GB VRAM, start with `yolov8n.pt`, image size `640`, batch size `4`, and data loader workers `0`. The app enables mixed-precision training automatically when GPU training is selected.

Keep at least 6 GB of free disk space before installing CUDA PyTorch. Install `requirements-gpu.txt` after `requirements.txt`, because a normal Ultralytics installation may install CPU-only PyTorch on Windows.

## MVP Scope

- Upload image files.
- Create multiclass labels.
- Draw bounding boxes on the selected image.
- Save complete or partial valid annotations through the FastAPI backend.
- Copy current boxes to selected images or all other images.
- Train a YOLO detection model from saved annotations.
- Auto annotate selected images or all images with the trained YOLO model.
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
