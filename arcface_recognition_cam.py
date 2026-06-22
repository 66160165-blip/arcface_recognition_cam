import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
import cv2
import numpy as np
import threading
import queue
import time
import argparse
from insightface.app import FaceAnalysis
from collections import deque

cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1") or 1))

# ==========================================
# ARGUMENT
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--rtsp", type=str,
    default=os.environ.get("CAMERA_RTSP_URL", ""))
parser.add_argument("--width",  type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default=os.environ.get("FACE_AI_PROVIDER", "auto").lower())
parser.add_argument("--det-size", type=int, default=int(os.environ.get("FACE_AI_DET_SIZE", "320") or 320))
parser.add_argument("--triple-detect", action="store_true", default=os.environ.get("FACE_AI_TRIPLE_DETECT", "false").lower() == "true")
args = parser.parse_args()
if not args.rtsp:
    parser.error("RTSP URL is required. Pass --rtsp or set CAMERA_RTSP_URL.")

# ==========================================
# CONFIG
# ==========================================

# --- Recognition ---
RECOG_THRESHOLD  = 0.28   # กล้องสูง+ไกล sim~0.31-0.36 → ต้องต่ำพอ
STRONG_THRESHOLD = 0.42
SIM_FLOOR        = 0.23   # 0.11 = false detect ชัดเจน → ตัดทิ้ง

# --- ระยะใบหน้า ---
NEAR_FACE_WIDTH = 35   # กล้องสูง หน้าจะเล็กกว่าปกติ

# --- Detection ---
DETECT_EVERY = 2
MIN_FACE     = 13     # ⬆ เพิ่มกลับ — face_w=11px = false detection
IOU_THRESH   = 0.15

# --- Upscale ก่อน detect ---
# กล้องสูง 5-8m → ใบหน้า ~10-20px → upscale frame ก่อน
# ทำให้ InsightFace detect ได้ดีขึ้นมาก
UPSCALE_DETECT = True    # เปิด upscale
UPSCALE_FACTOR = 1.8     # ขยาย 1.8x ก่อนส่ง detect
                          # (ไม่ควรเกิน 2.0 หรือ CPU จะช้า)

# --- Tracking ---
MAX_MISS        = 60
LOCK_THRESHOLD  = 3
UNLOCK_MISS_REQ = 80
NAME_VOTE_WINDOW    = 10
NAME_VOTE_MIN_COUNT = 3

# --- EMA ---
EMB_SMOOTH_ALPHA      = 0.6
EMB_UPDATE_MIN_SIM    = 0.28
EMB_UPDATE_MIN_FACE_W = 12    # ลดลง (กล้องสูง หน้าเล็กกว่าปกติ)

# --- แสดงกรอบ ---
# True  = แสดงเฉพาะคนที่จำได้ (Known เท่านั้น)
# False = แสดงทั้ง Known + Unknown
SHOW_KNOWN_ONLY = False  # False = แสดงทั้ง Known + Unknown

# สี (BGR)
COLOR_KNOWN   = (0, 220,   0)
COLOR_UNKNOWN = (0,   0, 255)

# ==========================================
# LOAD DATABASE & MODELS
# ==========================================
known_embeddings_raw = np.load("arcface_embeddings.npy", allow_pickle=True)
known_names          = np.load("arcface_names.npy",      allow_pickle=True)
known_embeddings     = known_embeddings_raw / np.linalg.norm(
    known_embeddings_raw, axis=1, keepdims=True)

print(f"✔ ฐานข้อมูล: {len(set(known_names))} คน | {len(known_names)} embeddings")
print(f"   ชื่อ: {list(set(known_names))}")

_dll_directory_handles = []


def add_nvidia_dll_directories():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    try:
        import site
        site_roots = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            site_roots.append(user_site)
    except Exception:
        site_roots = []

    added = []
    for root in site_roots:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for current, _, _ in os.walk(nvidia_root):
            if os.path.basename(current).lower() != "bin":
                continue
            try:
                _dll_directory_handles.append(os.add_dll_directory(current))
                path_parts = os.environ.get("PATH", "").split(os.pathsep)
                if current not in path_parts:
                    os.environ["PATH"] = current + os.pathsep + os.environ.get("PATH", "")
                added.append(current)
            except OSError:
                pass
    return added


def available_onnx_providers():
    try:
        added_dll_dirs = add_nvidia_dll_directories()
        if added_dll_dirs:
            print(f"[AI] added NVIDIA DLL directories: {added_dll_dirs}")
        import onnxruntime as ort
        if os.environ.get("FACE_AI_PRELOAD_DLLS", "true").lower() == "true" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception as preload_exc:
                print(f"[AI] ONNX Runtime preload_dlls warning: {preload_exc}")
        return ort.get_available_providers()
    except Exception as exc:
        print(f"[AI] cannot inspect ONNX Runtime providers: {exc}")
        return []


def choose_runtime(provider_name):
    available = available_onnx_providers()
    wants_cuda = provider_name == "cuda" or (provider_name == "auto" and "CUDAExecutionProvider" in available)
    if wants_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, available
    if provider_name == "cuda":
        print(f"[AI] CUDAExecutionProvider not available, fallback to CPU. Available providers: {available}")
    return ["CPUExecutionProvider"], -1, available


def make_face_app(det_size):
    app = FaceAnalysis(name="buffalo_l", providers=active_providers)
    app.prepare(ctx_id=active_ctx_id, det_size=(det_size, det_size))
    return app


active_providers, active_ctx_id, onnx_providers = choose_runtime(args.provider)
det_size = max(min(args.det_size, 640), 160)
try:
    if args.triple_detect:
        face_apps = [make_face_app(640), make_face_app(480), make_face_app(320)]
        mode_label = "Triple detection: 640+480+320"
    else:
        face_apps = [make_face_app(det_size)]
        mode_label = f"Fast single detection: {det_size}"
except Exception as exc:
    if active_providers[0] != "CUDAExecutionProvider":
        raise
    print(f"[AI] CUDA provider failed ({exc}), fallback to CPU")
    active_providers = ["CPUExecutionProvider"]
    active_ctx_id = -1
    if args.triple_detect:
        face_apps = [make_face_app(640), make_face_app(480), make_face_app(320)]
        mode_label = "Triple detection CPU fallback: 640+480+320"
    else:
        face_apps = [make_face_app(det_size)]
        mode_label = f"Fast single detection CPU fallback: {det_size}"

print(f"✔ โหลดโมเดลสำเร็จ ({mode_label}) providers={active_providers} available={onnx_providers} opencv_threads={cv2.getNumThreads()}")

# ==========================================
# RECOGNITION
# ==========================================
def recognize_face(face_embedding):
    norm = np.linalg.norm(face_embedding)
    if norm == 0:
        return "Unknown", 0.0, "NONE"
    face_norm = face_embedding / norm
    sims      = np.dot(known_embeddings, face_norm)
    idx       = int(np.argmax(sims))
    sim       = float(sims[idx])
    if sim < SIM_FLOOR:
        return "Unknown", sim, "SKIP"
    if sim >= STRONG_THRESHOLD:
        return known_names[idx], sim, "STRONG"
    elif sim >= RECOG_THRESHOLD:
        return known_names[idx], sim, "WEAK"
    else:
        return "Unknown", sim, "NONE"

# ==========================================
# DISPLAY DECISION
# ==========================================
def decide_display(name, sim, level, face_w, track):
    # Locked → คงชื่อไว้เสมอ
    if track.get("locked"):
        if level == "STRONG":
            return name
        return track["name"]

    track_known = track["name"] not in ("Unknown", "")

    if level == "SKIP":
        return track["name"] if track_known else "Unknown"

    if face_w >= NEAR_FACE_WIDTH:
        if level in ("STRONG", "WEAK"):
            return name
        return track["name"] if track_known else "Unknown"
    else:
        if level == "STRONG":
            return name
        elif level == "WEAK":
            if sim >= (STRONG_THRESHOLD - 0.05):
                return name
            return track["name"] if track_known else "Unknown"
        else:
            return track["name"] if track_known else "Unknown"

# ==========================================
# TRACKING HELPERS
# ==========================================
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    return inter / float(
        (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

def make_track(tid, bbox, emb, name, sim):
    return {
        "id":               tid,
        "bbox":             bbox,
        "embedding":        emb,
        "best_embedding":   emb,
        "best_sim":         sim,
        "smooth_embedding": emb.copy(),
        "best_face_w":      0,
        "name":             name,
        "display_name":     name,
        "sim":              sim,
        "miss":             0,
        "consec_miss":      0,
        "matched":          True,
        "vote_history":     deque([name], maxlen=NAME_VOTE_WINDOW),
        "known_streak":     1 if name not in ("Unknown", "") else 0,
        "locked":           False,
        "unlock_miss":      0,
    }

def confirm_track_identity(track, name, sim):
    track["vote_history"].append(name)
    track["sim"] = sim
    is_real = name not in ("Unknown", "")

    if is_real:
        track["known_streak"] = track.get("known_streak", 0) + 1
    else:
        track["known_streak"] = max(0, track.get("known_streak", 0) - 1)

    if (track["known_streak"] >= LOCK_THRESHOLD
            and not track.get("locked")
            and track["name"] not in ("Unknown", "")):
        track["locked"]      = True
        track["unlock_miss"] = 0
        print(f"[LOCKED] Track {track['id']} → {track['name']} "
              f"(streak={track['known_streak']})")

    if track.get("locked"):
        if is_real:
            track["name"] = name
        return

    history     = list(track["vote_history"])
    counts      = {}
    for n in history:
        counts[n] = counts.get(n, 0) + 1
    real_counts = {k: v for k, v in counts.items()
                   if k not in ("Unknown", "")}
    if real_counts:
        best_real = max(real_counts, key=real_counts.get)
        if real_counts[best_real] >= NAME_VOTE_MIN_COUNT:
            track["name"] = best_real
            return
    best = max(counts, key=counts.get)
    if counts[best] >= NAME_VOTE_MIN_COUNT:
        if track["name"] in ("Unknown", ""):
            track["name"] = best

# ==========================================
# DETECTION HELPER
# ==========================================
def detect_faces(frame):
    """
    Face detection:
    1. ถ้า UPSCALE_DETECT=True → ขยาย frame ก่อน detect
       bbox ที่ได้จะถูก scale กลับมาขนาดจริง
    2. ใช้ single model เป็นค่าเริ่มต้นเพื่อลด lag หรือเปิด --triple-detect เมื่อต้องการละเอียดขึ้น
    """
    if UPSCALE_DETECT:
        h, w = frame.shape[:2]
        big = cv2.resize(frame,
                         (int(w * UPSCALE_FACTOR), int(h * UPSCALE_FACTOR)),
                         interpolation=cv2.INTER_CUBIC)
    else:
        big = frame

    all_raw = []
    for app in face_apps:
        all_raw.extend(app.get(big))

    # Scale bbox กลับมาขนาดเดิม + wrap เป็น dict ไม่แตะ object เดิม
    scale = 1.0 / UPSCALE_FACTOR if UPSCALE_DETECT else 1.0
    faces = []
    used  = []
    for f in sorted(all_raw,
                    key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                    reverse=True):
        b_orig = (f.bbox * scale).astype(int)
        if not any(iou(b_orig, u) > 0.4 for u in used):
            faces.append({"bbox": b_orig, "embedding": f.embedding})
            used.append(b_orig)

    return faces

# ==========================================
# RTSP CAPTURE
# ==========================================
frame_queue = queue.Queue(maxsize=2)

def create_capture():
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, 25)
    return cap

cap = create_capture()

def capture_frames():
    global cap
    fail_count = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret: raise Exception()
            fail_count = 0
            if frame_queue.full():
                try: frame_queue.get_nowait()
                except: pass
            frame_queue.put(frame)
        except:
            fail_count += 1
            print(f"[RTSP] Reconnect #{fail_count}")
            cap.release()
            time.sleep(min(fail_count, 5))
            cap = create_capture()

threading.Thread(target=capture_frames, daemon=True).start()

# ==========================================
# MAIN LOOP
# ==========================================
tracks      = []
next_id     = 0
frame_count = 0
last_faces  = []
prev_time   = time.time()
fps         = 0.0

DEBUG_INTERVAL  = 3.0
last_debug_time = 0.0

print(f"✔ เริ่มระบบ (กล้องสูง ระยะไกล)")
print(f"  RECOG={RECOG_THRESHOLD} | STRONG={STRONG_THRESHOLD} | FLOOR={SIM_FLOOR}")
print(f"  NEAR={NEAR_FACE_WIDTH}px | MIN_FACE={MIN_FACE}px")
print(f"  UPSCALE={UPSCALE_DETECT} x{UPSCALE_FACTOR}")
print(f"  SHOW_KNOWN_ONLY={SHOW_KNOWN_ONLY}")
print("  สี: เขียว=จำได้ [L]=Locked | กด ESC เพื่อออก")

while True:
    try:
        frame = frame_queue.get(timeout=3)
    except:
        print("[WARN] frame timeout")
        continue

    frame = cv2.resize(frame, (args.width, args.height))
    frame_count += 1
    now = time.time()

    # ── Detection ─────────────────────────────────────────────────
    if frame_count % DETECT_EVERY == 0:
        last_faces = detect_faces(frame)
    faces = last_faces

    # ── กรอง ──────────────────────────────────────────────────────
    detections = []
    used_boxes = []
    for face in faces:
        # รองรับทั้ง dict (จาก detect_faces) และ InsightFace object
        if isinstance(face, dict):
            box = face["bbox"].astype(int) if hasattr(face["bbox"], "astype") else np.array(face["bbox"], dtype=int)
            emb = face["embedding"]
        else:
            box = face.bbox.astype(int)
            emb = face.embedding
        w, h = box[2]-box[0], box[3]-box[1]
        if w < MIN_FACE or h < MIN_FACE: continue
        ratio = w / h if h > 0 else 0
        if ratio < 0.35 or ratio > 2.0: continue
        if any(iou(box, ub) > 0.5 for ub in used_boxes): continue
        used_boxes.append(box)
        detections.append({"bbox": box, "embedding": emb, "width": w})

    for t in tracks:
        t["matched"] = False

    show_debug = (now - last_debug_time) >= DEBUG_INTERVAL

    # ── Matching + Recognition ─────────────────────────────────────
    for det in detections:
        name, sim, level = recognize_face(det["embedding"])

        if level == "SKIP" and det["width"] < 15:
            continue

        if show_debug:
            print(f"[DEBUG] sim={sim:.3f} name={name} "
                  f"level={level} face_w={det['width']}px")
            last_debug_time = now
            show_debug      = False

        best_track, best_score = None, 0.0
        for t in tracks:
            ref_emb = t.get("best_embedding", t["embedding"])
            cos     = cosine_sim(det["embedding"], ref_emb)
            box     = iou(det["bbox"], t["bbox"])
            score   = 0.70 * cos + 0.30 * box
            if score > best_score:
                best_score, best_track = score, t

        cos_ok = (best_track is not None and
                  cosine_sim(det["embedding"],
                             best_track.get("best_embedding",
                                            best_track["embedding"])) > 0.30)

        if best_track and best_score > IOU_THRESH and cos_ok:
            best_track["bbox"]        = det["bbox"]
            best_track["embedding"]   = det["embedding"]
            best_track["miss"]        = 0
            best_track["consec_miss"] = 0
            best_track["matched"]     = True
            best_track["unlock_miss"] = 0
            best_track["best_face_w"] = max(
                best_track.get("best_face_w", 0), det["width"])

            # EMA — อัพเดทเฉพาะ frame ดี
            prev_smooth = best_track.get("smooth_embedding", det["embedding"])
            frame_good  = (sim >= EMB_UPDATE_MIN_SIM and
                           det["width"] >= EMB_UPDATE_MIN_FACE_W)
            if frame_good:
                new_smooth = (EMB_SMOOTH_ALPHA * det["embedding"] +
                              (1 - EMB_SMOOTH_ALPHA) * prev_smooth)
                ns = np.linalg.norm(new_smooth)
                if ns > 0:
                    new_smooth = new_smooth / ns * np.linalg.norm(det["embedding"])
                best_track["smooth_embedding"] = new_smooth
            else:
                new_smooth = prev_smooth

            s_name, s_sim, s_level = recognize_face(new_smooth)
            if frame_good and s_sim > best_track.get("best_sim", 0):
                best_track["best_embedding"] = new_smooth
                best_track["best_sim"]       = s_sim

            name, sim, level = s_name, s_sim, s_level

            confirm_track_identity(
                best_track,
                name if level not in ("NONE", "SKIP") else "Unknown",
                sim)
            best_track["display_name"] = decide_display(
                name, sim, level, det["width"], best_track)

        else:
            init_name = name if level not in ("NONE", "SKIP") else "Unknown"
            nt = make_track(next_id, det["bbox"], det["embedding"],
                            init_name, sim)
            nt["display_name"] = decide_display(name, sim, level, det["width"], nt)
            nt["best_face_w"]  = det["width"]

            for old_t in tracks:
                ref = old_t.get("best_embedding", old_t["embedding"])
                if cosine_sim(det["embedding"], ref) > 0.50:
                    nt["best_face_w"] = max(
                        nt["best_face_w"], old_t.get("best_face_w", 0))
                    if old_t.get("locked") and old_t["name"] not in ("Unknown", ""):
                        nt["name"]             = old_t["name"]
                        nt["display_name"]     = old_t["name"]
                        nt["locked"]           = True
                        nt["known_streak"]     = old_t.get("known_streak", LOCK_THRESHOLD)
                        nt["best_embedding"]   = old_t.get("best_embedding", old_t["embedding"])
                        nt["best_sim"]         = old_t.get("best_sim", 0)
                        nt["smooth_embedding"] = old_t.get("smooth_embedding", old_t["embedding"])
                        print(f"[INHERIT] Track {next_id} ← "
                              f"{old_t['name']} จาก Track {old_t['id']}")
                    break

            tracks.append(nt)
            next_id += 1

    # ── Unlock + ลบ ───────────────────────────────────────────────
    for t in tracks:
        if t["matched"]:
            continue
        if not t.get("locked"):
            t["miss"] += 1
        else:
            t["consec_miss"] = t.get("consec_miss", 0) + 1
            if t["consec_miss"] >= 3:
                t["unlock_miss"] = t.get("unlock_miss", 0) + 1
            if t["unlock_miss"] % 10 == 0 and t["unlock_miss"] > 0:
                print(f"[MISS] Track {t['id']} "
                      f"unlock_miss={t['unlock_miss']}/{UNLOCK_MISS_REQ}")
            if t["unlock_miss"] >= UNLOCK_MISS_REQ:
                old = t["name"]
                t["locked"]       = False
                t["known_streak"] = 0
                t["name"]         = "Unknown"
                t["display_name"] = "Unknown"
                t["vote_history"].clear()
                t["miss"]         = 0
                t["consec_miss"]  = 0
                t["best_face_w"]  = 0
                print(f"[UNLOCK] Track {t['id']} ({old} → รอยืนยันใหม่)")

    tracks = [t for t in tracks
              if t.get("locked") or t["miss"] < MAX_MISS]

    # ── Drawing ───────────────────────────────────────────────────
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        dname   = t.get("display_name", t["name"])
        is_known = dname not in ("Unknown", "")

        # ถ้า SHOW_KNOWN_ONLY → ข้าม Unknown
        if SHOW_KNOWN_ONLY and not is_known:
            continue

        color = COLOR_KNOWN if is_known else COLOR_UNKNOWN
        icon  = " [L]" if t.get("locked") else ""
        label = f"ID{t['id']} {dname}{icon} {t['sim']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx, ly = x1, max(y1 - th - 8, 0)
        cv2.rectangle(frame, (lx, ly), (lx+tw+6, ly+th+6), color, -1)
        cv2.putText(frame, label, (lx+3, ly+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    dt        = time.time() - prev_time
    fps       = 0.9*fps + 0.1*(1.0/dt) if dt > 0 else fps
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.rectangle(frame, (args.width-125, 15), (args.width-111, 29), COLOR_KNOWN, -1)
    cv2.putText(frame, "Known", (args.width-107, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_KNOWN, 1)
    cv2.rectangle(frame, (args.width-125, 37), (args.width-111, 51), COLOR_UNKNOWN, -1)
    cv2.putText(frame, "Unknown", (args.width-107, 49),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_UNKNOWN, 1)

    cv2.imshow("Face Recognition CCTV", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
