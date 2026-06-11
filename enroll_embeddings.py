"""
enroll_embeddings.py
====================
ถ่ายรูประยะไกล (4-7m) เพิ่ม embedding เข้าฐานข้อมูลโดยตรง
ใช้ flow เดิมของระบบ: dataset_raw → dataset_clean → embeddings.npy

วิธีใช้:
  python enroll_embeddings.py
  → ใส่ชื่อ (Student ID)
  → SPACE = ถ่ายรูป
  → ESC   = เสร็จ บันทึกฐานข้อมูลอัตโนมัติ

แนะนำถ่าย:
  - ระยะ 1-2m  ~5 รูป
  - ระยะ 3-4m  ~5 รูป  
  - ระยะ 5-7m  ~10 รูป  ← สำคัญที่สุด
  - หันซ้าย/ขวา 15-30 องศา ในแต่ละระยะ
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ==========================================
# CONFIG — ตรงกับระบบเดิม
# ==========================================
DATASET_RAW   = "dataset_raw"
DATASET_CLEAN = "dataset_clean"
EMB_FILE      = "arcface_embeddings.npy"
NAME_FILE     = "arcface_names.npy"

MIRROR_VIEW    = True    # เหมือน capture_dataset.py เดิม
ENABLE_SHARPEN = True    # เหมือน capture_dataset.py เดิม

CAM_WIDTH  = 2560
CAM_HEIGHT = 1080
CAM_FPS    = 60

MIN_FACE_PX   = 15      # ขั้นต่ำ — ต่ำเพื่อ enroll ระยะไกลได้
MIN_SIM_DEDUP = 0.98    # dedup threshold — ไม่เพิ่ม embedding ที่เหมือนกัน 98%+

SCALE_FACTOR = 0.5      # resize เหมือน clean_dataset.py เดิม

# sharpen kernel เหมือนเดิม
SHARPEN_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

# ==========================================
# INPUT PERSON NAME
# ==========================================
person_name = input("Enter person name (Student ID): ").strip()
if not person_name:
    print("❌ ชื่อว่างเปล่า")
    exit()

# ==========================================
# SETUP FOLDERS
# ==========================================
raw_dir   = os.path.join(DATASET_RAW,   person_name)
clean_dir = os.path.join(DATASET_CLEAN, person_name)
os.makedirs(raw_dir,   exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

# หาเลขไฟล์ถัดไป (ต่อจากที่มีอยู่)
existing = [f for f in os.listdir(raw_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
numbers  = [int(os.path.splitext(f)[0])
            for f in existing if os.path.splitext(f)[0].isdigit()]
next_num = max(numbers) + 1 if numbers else 1

# ==========================================
# LOAD INSIGHTFACE
# ==========================================
print("🔄 โหลด InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✔ โหลดสำเร็จ")

# ==========================================
# LOAD EXISTING EMBEDDINGS
# ==========================================
if os.path.exists(EMB_FILE) and os.path.exists(NAME_FILE):
    existing_embs  = list(np.load(EMB_FILE,  allow_pickle=True))
    existing_names = list(np.load(NAME_FILE, allow_pickle=True))
    print(f"✔ ฐานข้อมูลเดิม: {len(set(existing_names))} คน | "
          f"{len(existing_names)} embeddings")
else:
    existing_embs  = []
    existing_names = []
    print("⚠️  ไม่พบฐานข้อมูลเดิม จะสร้างใหม่")

# backup ก่อนแก้ไข
if existing_embs:
    np.save(EMB_FILE  + ".bak", np.array(existing_embs))
    np.save(NAME_FILE + ".bak", np.array(existing_names))
    print("✔ Backup .bak สำเร็จ")

# normalize สำหรับ dedup
def norm_emb(e):
    n = np.linalg.norm(e)
    return e / n if n > 0 else e

existing_norms = [norm_emb(e) for e in existing_embs]

# ==========================================
# DEDUP CHECK
# ==========================================
def is_duplicate(new_norm):
    """เช็คว่า embedding ซ้ำกับที่มีอยู่ของคนนี้ไหม"""
    for i, e in enumerate(existing_norms):
        if existing_names[i] == person_name:
            if np.dot(e, new_norm) > MIN_SIM_DEDUP:
                return True
    return False

# ==========================================
# OPEN CAMERA
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 กล้อง: {aw}x{ah}")
print("=" * 60)
print(f"👤 ชื่อ: {person_name}")
print(f"📂 บันทึกที่: {raw_dir}")
print(f"🔢 เริ่มที่รูปที่: {next_num}")
print("⌨️  SPACE=ถ่ายรูป | ESC=เสร็จ+บันทึก DB")
print("=" * 60)

new_embs    = []   # embeddings ที่เพิ่มในเซสชันนี้
saved_count = 0

# ==========================================
# MAIN CAPTURE LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ อ่าน frame ไม่ได้")
        break

    if MIRROR_VIEW:
        frame = cv2.flip(frame, 1)
    if ENABLE_SHARPEN:
        frame = cv2.filter2D(frame, -1, SHARPEN_KERNEL)

    display = frame.copy()

    # ── detect realtime แสดงกรอบใบหน้า ──────────────────────────
    small = cv2.resize(display, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    faces = app.get(small)
    for face in faces:
        b = (face.bbox / SCALE_FACTOR).astype(int)
        fw = b[2] - b[0]
        cv2.rectangle(display, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(display, f"{fw}px", (b[0], b[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── UI ───────────────────────────────────────────────────────
    for text, y, color in [
        (f"Person: {person_name}", 40, (0, 255, 0)),
        (f"Saved this session: {saved_count}", 80, (0, 255, 255)),
        (f"Total in DB: {len(existing_embs)}", 120, (255, 255, 255)),
        (f"Res: {aw}x{ah}", 160, (255, 255, 255)),
        ("SPACE=Capture  ESC=Finish+Save DB", 200, (0, 200, 255)),
    ]:
        cv2.putText(display, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.imshow("Enroll Faces", display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC → เสร็จ
        break

    elif key == 32:   # SPACE → ถ่ายรูป
        # ── บันทึก raw ───────────────────────────────────────────
        raw_path = os.path.join(raw_dir, f"{next_num}.jpg")
        cv2.imwrite(raw_path, frame)

        # ── clean (resize เหมือน clean_dataset.py) ───────────────
        clean_img = cv2.resize(frame, None,
                                fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        clean_path = os.path.join(clean_dir, f"clean_{next_num}.jpg")
        cv2.imwrite(clean_path, clean_img)

        # ── extract embedding ─────────────────────────────────────
        det_faces = app.get(clean_img)

        # ถ้าไม่เจอ ลอง upscale x2 (ช่วยระยะไกล)
        if not det_faces:
            up = cv2.resize(clean_img, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)
            det_faces = app.get(up)

        if not det_faces:
            print(f"❌ [{next_num}] ไม่พบใบหน้า — ลองถ่ายใหม่ใกล้ขึ้นนิดนึง")
            next_num += 1
            saved_count += 1
            continue

        best_face = max(det_faces,
                        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        face_w = int(best_face.bbox[2] - best_face.bbox[0])

        if face_w < MIN_FACE_PX:
            print(f"⚠️  [{next_num}] ใบหน้าเล็กเกิน ({face_w}px) — บันทึกรูปไว้แต่ไม่เพิ่ม embedding")
            next_num += 1
            saved_count += 1
            continue

        emb      = best_face.embedding
        emb_norm = norm_emb(emb)

        # dedup check
        if is_duplicate(emb_norm):
            print(f"🔁 [{next_num}] Embedding ซ้ำกับที่มีอยู่ — บันทึกรูปแต่ข้าม embedding")
            next_num += 1
            saved_count += 1
            continue

        # เพิ่ม embedding
        new_embs.append(emb)
        existing_norms.append(emb_norm)   # เพิ่มเข้า pool สำหรับ dedup รูปถัดไป
        existing_names.append(person_name)

        print(f"✅ [{next_num}] face_w={face_w}px → เพิ่ม embedding สำเร็จ "
              f"(รวมเซสชันนี้: {len(new_embs)})")
        next_num    += 1
        saved_count += 1

# ==========================================
# SAVE DATABASE
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)

if not new_embs:
    print("⚠️  ไม่มี embedding ใหม่ — ฐานข้อมูลไม่เปลี่ยนแปลง")
else:
    # รวม existing จาก backup + new
    if os.path.exists(EMB_FILE + ".bak"):
        base_embs  = list(np.load(EMB_FILE  + ".bak", allow_pickle=True))
        base_names = list(np.load(NAME_FILE + ".bak", allow_pickle=True))
    else:
        base_embs, base_names = [], []

    final_embs  = base_embs  + new_embs
    final_names = base_names + [person_name] * len(new_embs)

    np.save(EMB_FILE,  np.array(final_embs))
    np.save(NAME_FILE, np.array(final_names))

    print(f"✅ บันทึกฐานข้อมูลแล้ว")
    print(f"   คน: {len(set(final_names))} คน")
    print(f"   Embeddings รวม: {len(final_names)}")
    print(f"   เพิ่มใหม่เซสชันนี้: {len(new_embs)}")
    print(f"\n✔ รัน arcface_recognition_cam.py ใหม่เพื่อโหลดฐานข้อมูลที่อัพเดท")

print(f"\n📸 รูปที่ถ่ายทั้งหมด: {saved_count} รูป")
print(f"   raw   → {raw_dir}/")
print(f"   clean → {clean_dir}/")
print("=" * 60)