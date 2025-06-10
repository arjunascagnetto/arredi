import argparse
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File


class DetectedElement(BaseModel):
    """Struttura dati per un elemento rilevato."""

    categoria: str
    file_path: str


def load_model(weights: str = "yolov8n-seg.pt") -> YOLO:
    """Carica il modello YOLOv8-segmentation su CPU."""

    return YOLO(weights)


def detect_elements(image: np.ndarray, model: YOLO, output_dir: Path) -> List[DetectedElement]:
    """Esegue la rilevazione e salva il crop di ogni oggetto trovato."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(image, device="cpu")
    elements: List[DetectedElement] = []
    for result in results:
        names = result.names
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            categoria = names[cls_id]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            crop = image[y1:y2, x1:x2]
            file_path = output_dir / f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(file_path), crop)
            elements.append(DetectedElement(categoria=categoria, file_path=str(file_path)))
    return elements


def run_cli(image_path: Path) -> None:
    """Esegue la pipeline da riga di comando."""

    model = load_model()
    image = cv2.imread(str(image_path))
    detections = detect_elements(image, model, Path("outputs/elements"))
    for det in detections:
        print(det.json())


def create_app() -> FastAPI:
    """Crea e restituisce una FastAPI app per l'inference via API."""

    app = FastAPI(title="YOLOv8 Segmentation")
    model = load_model()

    @app.post("/detect", response_model=List[DetectedElement])
    async def detect(file: UploadFile = File(...)) -> List[DetectedElement]:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        detections = detect_elements(image, model, Path("outputs/elements"))
        return detections

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rileva elementi di interni con YOLOv8-segmentation")
    parser.add_argument("image", type=Path, help="Percorso dell'immagine di input")
    args = parser.parse_args()
    run_cli(args.image)
