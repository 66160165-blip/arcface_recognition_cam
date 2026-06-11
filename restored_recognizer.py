# restored_recognizer.py
# ==========================================
# เขียนผลลง results/result_<track_id>.json แยกต่อ track
# → ไม่มี Race Condition กับ main cam อีกต่อไป
# ==========================================

import os
import cv2
import numpy as np
import time
import json
from insightface.app import FaceAnalysis

# ==========================================
# FOLDERS & FILES
# ==========================================
RESTORED_DIR = "restored_faces"
RESULT_DIR   = "results"          # ← เปลี่ยนจาก result.json เดียว → โฟลเดอร์

os.makedirs(RESTORED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR,   exist_ok=True)

# ==========================================
# LOAD DATABASE
# ==========================================
known_embeddings_raw = np.load("arcface_embeddings.npy", allow_pickle=True)
known_names          = np.load("arcface_names.npy",      allow_pickle=True)

known_embeddings = known_embeddings_raw / np.linalg.norm(
    known_embeddings_raw, axis=1, keepdims=True
)

print(f"✔ โหลดฐานข้อมูล: {len(set(known_names))} คน | {len(known_names)} embeddings")

# ==========================================
# LOAD MODEL
# ==========================================
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✔ โหลด InsightFace สำเร็จ")

# ==========================================
# CONFIG
# ==========================================
RECOG_THRESHOLD  = 0.50
STRONG_THRESHOLD = 0.62
MIN_FACE_SIZE    = 40

# ==========================================
# HELPERS
# ==========================================

def extract_meta(filename: str):
    """
    รูปแบบใหม่: track_<id>_<ts>_<x1>_<y1>_<x2>_<y2>_restored.png
    คืน (track_id, bbox) หรือ (None, None) ถ้า parse ไม่ได้
    """
    try:
        # ตัด _restored.png ออกก่อน แล้ว split
        base  = filename.replace("_restored.png", "").replace("_restored.jpg", "")
        parts = base.split("_")
        # parts: [track, id, ts_date, ts_time, x1, y1, x2, y2]
        # หรือถ้า ts เป็น YYYYMMDD_HHMMSS จะได้ 8 parts
        if len(parts) < 8 or parts[0] != "track":
            return None, None
        track_id = parts[1]
        x1, y1, x2, y2 = int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
        return track_id, (x1, y1, x2, y2)
    except Exception:
        return None, None


def write_result(track_id: str, name: str, sim: float, matched: bool):
    """
    เขียนผลลง results/result_<track_id>.json
    ไฟล์แยกต่อ track → main cam อ่าน+ลบทิ้งเองโดยไม่ชนใคร
    เขียนผ่าน .tmp แล้ว rename → atomic ทั้ง Windows และ Linux
    """
    path = os.path.join(RESULT_DIR, f"result_{track_id}.json")
    data = {
        "name":    name,
        "sim":     round(sim, 4),
        "matched": matched,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)   # atomic replace

    status = "✔ MATCH" if matched else "✘ NO MATCH"
    print(f"[AI RESULT] Track {track_id} → {name} (sim={sim:.3f})  {status}")
    print(f"            เขียนลง: {path}")


def recognize_face(face_embedding: np.ndarray):
    norm = np.linalg.norm(face_embedding)
    if norm == 0:
        return "Unknown", 0.0, False

    face_norm = face_embedding / norm
    sims      = np.dot(known_embeddings, face_norm)

    idx = int(np.argmax(sims))
    sim = float(sims[idx])

    if sim >= RECOG_THRESHOLD:
        return known_names[idx], sim, True
    else:
        return "Unknown", sim, False


# ==========================================
# MAIN LOOP
# ==========================================
print("✔ Restored Recognizer Started")
print(f"📂 Watching : {RESTORED_DIR}")
print(f"📤 Result   : {RESULT_DIR}/result_<track_id>.json")
print(f"   RECOG_THRESHOLD={RECOG_THRESHOLD} | STRONG_THRESHOLD={STRONG_THRESHOLD}")

while True:
    try:
        files = [
            f for f in os.listdir(RESTORED_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for filename in files:
            image_path = os.path.join(RESTORED_DIR, filename)

            track_id, bbox = extract_meta(filename)
            if track_id is None:
                print(f"⚠️  ชื่อไฟล์ไม่ตรงรูปแบบ ข้าม: {filename}")
                try: os.remove(image_path)
                except Exception: pass
                continue

            bx1, by1, bx2, by2 = bbox
            print(f"\n🔍 Processing: {filename}  [track_id={track_id} bbox=({bx1},{by1},{bx2},{by2})]")
            time.sleep(0.2)

            img = cv2.imread(image_path)
            if img is None:
                print("❌ อ่านรูปไม่ได้")
                try: os.remove(image_path)
                except Exception: pass
                continue

            h, w = img.shape[:2]

            # ── ลอง detect บน full frame ก่อน ─────────────────────────
            faces = app.get(img)

            if not faces:
                # ── fallback: crop บริเวณใบหน้าแล้ว detect ใหม่ ────────
                pad  = 30
                cx1  = max(0, bx1 - pad)
                cy1  = max(0, by1 - pad)
                cx2  = min(w,  bx2 + pad)
                cy2  = min(h,  by2 + pad)
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    # upscale crop ให้ใหญ่ขึ้น 2x ก่อน detect
                    crop_up = cv2.resize(crop, None, fx=2, fy=2,
                                         interpolation=cv2.INTER_CUBIC)
                    faces = app.get(crop_up)
                    print(f"   [CROP+UPSCALE] detect {len(faces)} faces บน crop")

            if not faces:
                print("❌ ไม่พบใบหน้าในรูป → Unknown")
                write_result(track_id, "Unknown", 0.0, False)
                try: os.remove(image_path)
                except Exception: pass
                continue

            best_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            face_w = int(best_face.bbox[2] - best_face.bbox[0])

            if face_w < MIN_FACE_SIZE:
                print(f"⚠️  ใบหน้าเล็กเกินไป ({face_w}px) → Unknown")
                write_result(track_id, "Unknown", 0.0, False)
                try: os.remove(image_path)
                except Exception: pass
                continue

            name, sim, matched = recognize_face(best_face.embedding)
            write_result(track_id, name, sim, matched)

            try: os.remove(image_path)
            except Exception as e:
                print(f"⚠️  ลบไฟล์ไม่ได้: {e}")

    except Exception as e:
        print("❌ Recognizer Error:", e)

    time.sleep(0.5)