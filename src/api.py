"""
HTTP API for the Deepfake Detector frontend.

This exposes the trained model pipeline to the Next.js interface.
"""

import cgi
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import models, transforms

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY_SIZE = 12 * 1024 * 1024
FAKE_THRESHOLD = 0.75
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

        # Arquitectura tem de ser igual à do treino (com Dropout)
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


class DetectorHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
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

            self.send_json(200, payload)
            return

        self.send_json(404, {"error": "Endpoint não encontrado."})

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, {"error": "Endpoint não encontrado."})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_json(400, {"error": "Pedido sem ficheiro."})
            return

        if content_length > MAX_BODY_SIZE:
            self.send_json(413, {"error": "A imagem deve ter no máximo 10MB."})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(415, {"error": "Envie a imagem como multipart/form-data."})
            return

        body = self.rfile.read(content_length)
        form = cgi.FieldStorage(
            fp=io.BytesIO(body),
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
        )

        file_item = form["file"] if "file" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            self.send_json(400, {"error": "Campo de ficheiro `file` não encontrado."})
            return

        try:
            image_bytes = file_item.file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = predict_image(image)
            print(
                "Predição:",
                f"REAL={result['probabilityReal']:.4f}",
                f"FAKE={result['probabilityFake']:.4f}",
                f"classe={result['prediction']}",
                f"raw={result['rawProbabilities']}",
            )
            self.send_json(200, result)
        except ValueError as error:
            self.send_json(422, {"error": str(error)})
        except FileNotFoundError:
            self.send_json(500, {"error": f"Ficheiro `{MODEL_PATH.name}` não encontrado."})
        except Exception as error:
            self.send_json(500, {"error": f"Erro ao analisar imagem: {error}"})


def main():
    load_detector()
    try:
        server = ThreadingHTTPServer((HOST, PORT), DetectorHandler)
    except OSError as error:
        if error.errno == 48:
            print(f"A porta {PORT} já está em uso.")
            print(f"Se http://localhost:{PORT}/health responder, a API já está ligada.")
            print("Para usar outra porta: PORT=8001 npm run api")
            return
        raise

    print(f"Deepfake Detector API pronta em http://{HOST}:{PORT}")
    print("Endpoint: POST /predict com multipart field `file`")
    server.serve_forever()


if __name__ == "__main__":
    main()
