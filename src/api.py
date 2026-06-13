import io
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY_SIZE = 12 * 1024 * 1024
FAKE_THRESHOLD = 0.50
FAKE_CLASS_INDEX = int(os.environ.get("FAKE_CLASS_INDEX", "1"))
REAL_CLASS_INDEX = 1 - FAKE_CLASS_INDEX
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "best_model.pth"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT_DIR / MODEL_PATH

if FAKE_CLASS_INDEX not in (0, 1):
    raise ValueError("FAKE_CLASS_INDEX deve ser 0 ou 1.")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_MODEL = None
_MTCNN = None
_DEVICE = None
_MODEL_SIGNATURE = None
_MODEL_LOCK = Lock()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def get_model_signature():
    model_stat = MODEL_PATH.stat()
    return model_stat.st_mtime_ns, model_stat.st_size


def model_info(signature=None):
    if signature is None:
        signature = get_model_signature()

    modified_ns, size_bytes = signature
    return {
        "modelPath": str(MODEL_PATH),
        "modelModifiedNs": modified_ns,
        "modelSizeBytes": size_bytes,
    }


def load_detector():
    global _MODEL, _MTCNN, _DEVICE, _MODEL_SIGNATURE

    current_signature = get_model_signature()
    if (
        _MODEL is not None
        and _MTCNN is not None
        and _MODEL_SIGNATURE == current_signature
    ):
        return _MODEL, _MTCNN, _DEVICE

    with _MODEL_LOCK:
        current_signature = get_model_signature()
        if (
            _MODEL is not None
            and _MTCNN is not None
            and _MODEL_SIGNATURE == current_signature
        ):
            return _MODEL, _MTCNN, _DEVICE

        _DEVICE = get_device()

        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, 2)
        )

        print(f"A carregar modelo: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(_DEVICE)
        model.eval()

        if _MTCNN is None:
            _MTCNN = MTCNN(
                image_size=224,
                margin=40,
                keep_all=False,
                post_process=False,
                device="cpu",
                select_largest=True,
            )

        _MODEL = model
        _MODEL_SIGNATURE = current_signature
        return _MODEL, _MTCNN, _DEVICE


def predict_image(image):
    model, mtcnn, device = load_detector()

    face_tensor = mtcnn(image)
    if face_tensor is None:
        raise ValueError("Não foi detetada nenhuma face na imagem.")

    face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    face_pil = Image.fromarray(face_np)
    input_tensor = eval_transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prob_fake = float(probs[0, FAKE_CLASS_INDEX])
        prob_real = float(probs[0, REAL_CLASS_INDEX])

    is_fake = prob_fake >= FAKE_THRESHOLD

    return {
        "probabilityFake": prob_fake,
        "probabilityReal": prob_real,
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": prob_fake if is_fake else prob_real,
        "threshold": FAKE_THRESHOLD,
        "decisionRule": "threshold",
        "fakeClassIndex": FAKE_CLASS_INDEX,
        "realClassIndex": REAL_CLASS_INDEX,
        "rawLogits": [float(value) for value in output[0].cpu()],
        "rawProbabilities": [float(value) for value in probs[0].cpu()],
        "device": str(device),
        "mode": "model",
        **model_info(_MODEL_SIGNATURE),
    }


@app.on_event("startup")
def startup():
    load_detector()


@app.get("/health")
def health():
    payload = {
        "ok": True,
        "loaded": _MODEL is not None,
        "fakeClassIndex": FAKE_CLASS_INDEX,
        "realClassIndex": REAL_CLASS_INDEX,
        "loadedModel": model_info(_MODEL_SIGNATURE) if _MODEL_SIGNATURE else None,
    }
    try:
        payload["currentModel"] = model_info()
    except FileNotFoundError:
        payload["ok"] = False
        payload["error"] = f"Ficheiro `{MODEL_PATH.name}` não encontrado."
    return payload


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    if len(contents) > MAX_BODY_SIZE:
        raise HTTPException(status_code=413, detail="A imagem deve ter no máximo 10MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(image)
        print(
            "Predição:",
            f"REAL={result['probabilityReal']:.4f}",
            f"FAKE={result['probabilityFake']:.4f}",
            f"classe={result['prediction']}",
            f"raw={result['rawProbabilities']}",
        )
        return result
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Ficheiro `{MODEL_PATH.name}` não encontrado.")
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {error}")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
