from flask import Flask, request
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import requests
import time

app = Flask(__name__)

# =====================
# CONFIG
# =====================
PI_URL = "http://10.80.83.81:5000/unlock"

RECOG_THRESHOLD = 0.5
STRONG_THRESHOLD = 0.65

last_unlock = 0
COOLDOWN = 5

# =====================
# LOAD MODEL
# =====================
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(224, 224))

# =====================
# LOAD DB
# =====================
embeddings = np.load("arcface_embeddings.npy", allow_pickle=True)
names = np.load("arcface_names.npy", allow_pickle=True)

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("✔ AI SERVER READY")

# =====================
# RECOGNITION
# =====================
def recognize(embedding):
    emb = embedding / np.linalg.norm(embedding)
    sims = np.dot(embeddings, emb)

    idx = np.argmax(sims)
    sim = sims[idx]

    if sim >= STRONG_THRESHOLD:
        return names[idx], sim
    elif sim >= RECOG_THRESHOLD:
        return names[idx], sim
    else:
        return "Unknown", sim

# =====================
# UNLOCK
# =====================
def unlock(name):
    global last_unlock

    now = time.time()
    if now - last_unlock < COOLDOWN:
        return

    try:
        requests.get(PI_URL, timeout=1)
        print(f"🚪 เปิดประตู → {name}")
        last_unlock = now
    except Exception as e:
        print("❌ Pi error:", e)

# =====================
# RECEIVE FRAME
# =====================
@app.route("/frame", methods=["POST"])
def receive_frame():
    file = request.files["frame"]

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (480, 360))

    faces = face_app.get(frame)

    for face in faces:
        name, sim = recognize(face.embedding)

        print(f"[AI] {name} {sim:.2f}")

        if name != "Unknown":
            unlock(name)

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)