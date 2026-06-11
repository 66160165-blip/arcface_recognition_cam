from flask import Flask, request, jsonify
import numpy as np
import cv2
import requests
from insightface.app import FaceAnalysis
import time
import threading
from collections import deque

app = Flask(__name__)

# =========================
# CONFIG
# =========================
PI_URL = "http://10.80.83.81:5000/unlock"

RECOG_THRESHOLD  = 0.50
STRONG_THRESHOLD = 0.65

UNLOCK_COOLDOWN  = 5      # วินาที ห้ามเปิดซ้ำภายใน
STABLE_REQUIRED  = 3      # ต้องเจอชื่อเดิมกี่ครั้งติดกันก่อนเปิด
REQUEST_TIMEOUT  = 5

# =========================
# LOAD MODEL
# =========================
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(320, 320))

# =========================
# LOAD DATABASE
# =========================
known_embeddings = np.load("arcface_embeddings.npy", allow_pickle=True)
known_names      = np.load("arcface_names.npy",      allow_pickle=True)

known_embeddings = known_embeddings / np.linalg.norm(
    known_embeddings, axis=1, keepdims=True
)

print("✔ AI SERVER READY")

# =========================
# UNLOCK WORKER (non-blocking)
# ส่ง HTTP ไป Pi ใน background thread
# Flask handler ไม่ต้องรอ
# =========================
class UnlockWorker:
    def __init__(self):
        self._last_time  = 0
        self._lock       = threading.Lock()   # thread-safe cooldown
        self._queue      = deque()
        self._thread     = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def request_unlock(self, name: str) -> bool:
        """
        เรียกจาก Flask handler (หลาย thread พร้อมกันได้)
        return True ถ้ารับคำสั่งไว้ได้, False ถ้ายัง cooldown
        """
        now = time.time()
        with self._lock:
            if now - self._last_time < UNLOCK_COOLDOWN:
                return False
            self._last_time = now   # จองเวลาทันที ไม่รอ response

        self._queue.append(name)
        return True

    def _worker(self):
        while True:
            if self._queue:
                name = self._queue.popleft()
                try:
                    print(f"🚪 ส่งคำสั่งเปิดประตู → {name}")
                    res = requests.get(PI_URL, timeout=REQUEST_TIMEOUT)
                    print(f"[PI] status={res.status_code} | {res.text[:80]}")
                except Exception as e:
                    print(f"❌ Pi error: {e}")
            else:
                time.sleep(0.05)


# =========================
# STABLE COUNTER (per-client)
# ป้องกัน false-positive จาก 1 frame
# ใช้ lock เพราะ Flask อาจ multi-thread
# =========================
class StableCounter:
    def __init__(self):
        self._lock      = threading.Lock()
        self._last_name = None
        self._count     = 0

    def update(self, name: str) -> int:
        """อัปเดต และคืนค่า stable count ปัจจุบัน"""
        with self._lock:
            if name == self._last_name and name != "Unknown":
                self._count += 1
            else:
                self._last_name = name
                self._count     = 1
            return self._count

    def reset(self):
        with self._lock:
            self._count = 0


unlocker = UnlockWorker()
stable   = StableCounter()

# =========================
# RECOGNITION
# =========================
def recognize(embedding: np.ndarray):
    emb  = embedding / np.linalg.norm(embedding)
    sims = np.dot(known_embeddings, emb)
    idx  = int(np.argmax(sims))
    sim  = float(sims[idx])

    if sim >= STRONG_THRESHOLD:
        return known_names[idx], sim, "STRONG"
    elif sim >= RECOG_THRESHOLD:
        return known_names[idx], sim, "WEAK"
    else:
        return "Unknown", sim, "NONE"

# =========================
# API  /frame
# =========================
@app.route('/frame', methods=['POST'])
def receive_frame():
    # --- decode ภาพ ---
    data  = request.data
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "decode_failed"}), 400

    # --- detect + recognize ---
    faces = face_app.get(frame)

    best_name  = "Unknown"
    best_sim   = 0.0
    best_level = "NONE"
    unlocked   = False

    for face in faces:
        name, sim, level = recognize(face.embedding)

        # เลือก face ที่ sim สูงสุด (กรณีมีหลายคนในเฟรม)
        if sim > best_sim:
            best_name  = name
            best_sim   = sim
            best_level = level

    # --- stable + unlock ---
    count = stable.update(best_name)

    print(f"[AI] {best_name} ({best_sim:.2f}) level={best_level} stable={count}")

    if best_level == "STRONG" and count >= STABLE_REQUIRED:
        unlocked = unlocker.request_unlock(best_name)
        if unlocked:
            stable.reset()

    return jsonify({
        "name"    : best_name,
        "sim"     : round(best_sim, 4),
        "level"   : best_level,
        "stable"  : count,
        "unlocked": unlocked,
    })

# =========================
# HEALTH CHECK
# =========================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "buffalo_l"})

# =========================
# RUN  — threaded=True รับหลาย request พร้อมกัน
# =========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, threaded=True)