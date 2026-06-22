import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("FACE_AI_CPU_THREADS", "4"))
import threading
import time
import zlib
from collections import deque

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, request
from insightface.app import FaceAnalysis

cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "1") or 1))


parser = argparse.ArgumentParser()
parser.add_argument("--rtsp", type=str, default="", help="Optional RTSP URL for IP camera")
parser.add_argument("--width", type=int, default=640, help="Resize RTSP frame width before recognition")
parser.add_argument("--height", type=int, default=360, help="Resize RTSP frame height before recognition")
parser.add_argument("--interval", type=float, default=2.0, help="Seconds between RTSP recognition runs")
parser.add_argument("--preview-fps", type=float, default=float(os.environ.get("CAMERA_PREVIEW_FPS", "8") or 8), help="Target FPS for lightweight camera preview frames")
parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("CAMERA_JPEG_QUALITY", "78") or 78), help="JPEG quality for preview frames")
parser.add_argument("--frame-stale-seconds", type=float, default=float(os.environ.get("CAMERA_FRAME_STALE_SECONDS", "2.5") or 2.5), help="Seconds before an unchanged RTSP frame is treated as stale")
parser.add_argument("--zone", type=str, default=os.environ.get("CAMERA_ZONE_NAME", ""), help="Cabinet/return zone name for this camera")
parser.add_argument("--marker", type=str, default=os.environ.get("CAMERA_MARKER_CODE", ""), help="Shelf marker code currently visible to this camera")
parser.add_argument("--object-placement", type=str, default=os.environ.get("CAMERA_OBJECT_PLACEMENT", ""), help="Optional object placement status: detected, uncertain, not_detected")
parser.add_argument("--object-confidence", type=float, default=float(os.environ.get("CAMERA_OBJECT_CONFIDENCE", "0") or 0), help="Optional object placement confidence score")
parser.add_argument("--unlock-url", type=str, default=os.environ.get("PI_UNLOCK_URL", "http://10.80.83.81:5000/unlock"), help="Optional hardware unlock URL")
parser.add_argument("--unlock-enabled", action="store_true", default=os.environ.get("PI_UNLOCK_ENABLED", "false").lower() == "true", help="Enable hardware unlock requests after strong face recognition")
parser.add_argument("--provider", choices=["auto", "cuda", "cpu"], default=os.environ.get("FACE_AI_PROVIDER", "cpu").lower(), help="Inference provider: auto, cuda, or cpu")
parser.add_argument("--det-size", type=int, default=int(os.environ.get("FACE_AI_DET_SIZE", "320") or 320), help="InsightFace detector size")
parser.add_argument("--port", type=int, default=5001)
args = parser.parse_args()

app = Flask(__name__)

PI_URL = args.unlock_url
RECOG_THRESHOLD = 0.50
STRONG_THRESHOLD = 0.65
UNLOCK_COOLDOWN = 5
STABLE_REQUIRED = 3
REQUEST_TIMEOUT = 5


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


def load_face_app():
    providers, ctx_id, available = choose_runtime(args.provider)
    det_size = max(min(args.det_size, 640), 160)
    try:
        app_instance = FaceAnalysis(name="buffalo_l", providers=providers)
        app_instance.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        session_providers = get_model_session_providers(app_instance)
        if providers[0] == "CUDAExecutionProvider" and not model_sessions_use_cuda(session_providers):
            print(f"[AI] CUDA provider listed but model sessions are not using CUDA: {session_providers}. Fallback to CPU.")
            providers = ["CPUExecutionProvider"]
            app_instance = FaceAnalysis(name="buffalo_l", providers=providers)
            app_instance.prepare(ctx_id=-1, det_size=(det_size, det_size))
            session_providers = get_model_session_providers(app_instance)
        return app_instance, providers, available, det_size, session_providers
    except Exception as exc:
        if providers[0] == "CUDAExecutionProvider":
            print(f"[AI] CUDA provider failed ({exc}), fallback to CPU")
            providers = ["CPUExecutionProvider"]
            app_instance = FaceAnalysis(name="buffalo_l", providers=providers)
            app_instance.prepare(ctx_id=-1, det_size=(det_size, det_size))
            return app_instance, providers, available, det_size, get_model_session_providers(app_instance)
        raise


def get_model_session_providers(app_instance):
    session_providers = {}
    for name, model in getattr(app_instance, "models", {}).items():
        session = getattr(model, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            session_providers[name] = session.get_providers()
    return session_providers


def model_sessions_use_cuda(session_providers):
    return any("CUDAExecutionProvider" in providers for providers in session_providers.values())


face_app, active_providers, onnx_providers, active_det_size, model_session_providers = load_face_app()
print(f"[AI] providers={active_providers} model_sessions={model_session_providers} available={onnx_providers} det_size={active_det_size} opencv_threads={cv2.getNumThreads()}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
known_embeddings = np.load(os.path.join(BASE_DIR, "arcface_embeddings.npy"), allow_pickle=True)
known_names = np.load(os.path.join(BASE_DIR, "arcface_names.npy"), allow_pickle=True)
known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)

print("AI SERVER READY")


class UnlockWorker:
    def __init__(self):
        self._last_time = 0
        self._lock = threading.Lock()
        self._queue = deque()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def request_unlock(self, name: str) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_time < UNLOCK_COOLDOWN:
                return False
            self._last_time = now
        self._queue.append(name)
        return True

    def _worker(self):
        while True:
            if self._queue:
                name = self._queue.popleft()
                try:
                    res = requests.get(PI_URL, timeout=REQUEST_TIMEOUT)
                    print(f"[PI] unlock for {name}: status={res.status_code} {res.text[:80]}")
                except Exception as exc:
                    print(f"[PI] error: {exc}")
            else:
                time.sleep(0.05)


class StableCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_name = None
        self._count = 0

    def update(self, name: str) -> int:
        with self._lock:
            if name == self._last_name and name != "Unknown":
                self._count += 1
            else:
                self._last_name = name
                self._count = 1
            return self._count

    def reset(self):
        with self._lock:
            self._count = 0


unlocker = UnlockWorker()
browser_stable = StableCounter()
rtsp_stable = StableCounter()
latest_camera_lock = threading.Lock()
latest_frame_lock = threading.Lock()
latest_frame_jpeg = None
latest_frame_bgr = None
latest_frame_updated_at = None
latest_frame_signature = None
latest_camera_result = {
    "name": "Unknown",
    "sim": 0.0,
    "level": "NONE",
    "stable": 0,
    "unlocked": False,
    "source": "rtsp",
    "zone": args.zone or None,
    "zoneSource": "configured" if args.zone else None,
    "marker": args.marker or None,
    "markerSource": "configured" if args.marker else None,
    "objectPlacementStatus": args.object_placement or None,
    "objectPlacementSource": "configured" if args.object_placement else None,
    "objectPlacementConfidence": args.object_confidence,
    "updatedAt": None,
    "error": "rtsp_not_configured",
}


def frame_age_seconds(updated_at=None):
    if not updated_at:
        return None
    return max(time.time() - updated_at, 0)


def frame_is_stale(updated_at=None):
    age = frame_age_seconds(updated_at)
    return age is None or age > max(args.frame_stale_seconds, 0.5)


def camera_base_result(error="camera_frame_stale", updated_at=None):
    age = frame_age_seconds(updated_at)
    return {
        "name": "Unknown",
        "sim": 0.0,
        "level": "NONE",
        "stable": 0,
        "unlocked": False,
        "source": "rtsp",
        "zone": args.zone or None,
        "zoneSource": "configured" if args.zone else None,
        "marker": args.marker or None,
        "markerSource": "configured" if args.marker else None,
        "objectPlacementStatus": args.object_placement or None,
        "objectPlacementSource": "configured" if args.object_placement else None,
        "objectPlacementConfidence": args.object_confidence,
        "updatedAt": time.time(),
        "frameUpdatedAt": updated_at,
        "frameAgeSeconds": round(age, 2) if age is not None else None,
        "frameReady": False,
        "error": error,
    }


def frame_signature(frame):
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    return zlib.crc32(small.tobytes())


def recognize(embedding: np.ndarray):
    emb = embedding / np.linalg.norm(embedding)
    sims = np.dot(known_embeddings, emb)
    idx = int(np.argmax(sims))
    sim = float(sims[idx])

    if sim >= STRONG_THRESHOLD:
        return str(known_names[idx]), sim, "STRONG"
    if sim >= RECOG_THRESHOLD:
        return str(known_names[idx]), sim, "WEAK"
    return "Unknown", sim, "NONE"


def recognize_frame(frame, stable_counter, allow_unlock=False):
    faces = face_app.get(frame)
    best_name = "Unknown"
    best_sim = 0.0
    best_level = "NONE"

    for face in faces:
        name, sim, level = recognize(face.embedding)
        if sim > best_sim:
            best_name = name
            best_sim = sim
            best_level = level

    count = stable_counter.update(best_name)
    unlocked = False
    if allow_unlock and args.unlock_enabled and best_level == "STRONG" and count >= STABLE_REQUIRED:
        unlocked = unlocker.request_unlock(best_name)
        if unlocked:
            stable_counter.reset()

    return {
        "name": best_name,
        "sim": round(best_sim, 4),
        "level": best_level,
        "stable": count,
        "unlocked": unlocked,
    }


def rtsp_worker():
    global latest_camera_result, latest_frame_bgr, latest_frame_jpeg, latest_frame_updated_at, latest_frame_signature
    if not args.rtsp:
        return

    print("[RTSP] opening configured stream")
    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fail_count = 0
    preview_delay = 1 / max(args.preview_fps, 1)
    jpeg_quality = min(max(args.jpeg_quality, 35), 95)

    while True:
        ok, frame = cap.read()
        if not ok:
            fail_count += 1
            rtsp_stable.reset()
            with latest_frame_lock:
                latest_frame_bgr = None
                latest_frame_jpeg = None
                latest_frame_updated_at = None
                latest_frame_signature = None
            with latest_camera_lock:
                latest_camera_result = camera_base_result(error=f"rtsp_read_failed_{fail_count}")
            cap.release()
            time.sleep(min(fail_count, 5))
            cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        fail_count = 0
        if args.width and args.height:
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        encoded_ok, encoded_frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if encoded_ok:
            signature = frame_signature(frame)
            with latest_frame_lock:
                latest_frame_bgr = frame.copy()
                latest_frame_jpeg = encoded_frame.tobytes()
                if signature != latest_frame_signature:
                    latest_frame_signature = signature
                    latest_frame_updated_at = time.time()

        time.sleep(preview_delay)


def rtsp_recognition_worker():
    global latest_camera_result
    if not args.rtsp:
        return

    while True:
        with latest_frame_lock:
            frame = latest_frame_bgr.copy() if latest_frame_bgr is not None else None
            frame_updated_at = latest_frame_updated_at

        if frame is None or frame_is_stale(frame_updated_at):
            rtsp_stable.reset()
            with latest_camera_lock:
                latest_camera_result = camera_base_result(
                    error="camera_frame_stale" if frame is not None else "frame_not_ready",
                    updated_at=frame_updated_at,
                )
            time.sleep(0.2)
            continue

        try:
            result = recognize_frame(frame, rtsp_stable, allow_unlock=True)
        except Exception as exc:
            rtsp_stable.reset()
            print(f"[RTSP AI] inference failed: {type(exc).__name__}: {exc}")
            with latest_camera_lock:
                latest_camera_result = camera_base_result(
                    error="inference_failed",
                    updated_at=frame_updated_at,
                )
            time.sleep(max(args.interval, 0.5))
            continue
        age = frame_age_seconds(frame_updated_at)
        result.update({
            "source": "rtsp",
            "zone": args.zone or None,
            "zoneSource": "configured" if args.zone else None,
            "marker": args.marker or None,
            "markerSource": "configured" if args.marker else None,
            "objectPlacementStatus": args.object_placement or None,
            "objectPlacementSource": "configured" if args.object_placement else None,
            "objectPlacementConfidence": args.object_confidence,
            "updatedAt": time.time(),
            "frameUpdatedAt": frame_updated_at,
            "frameAgeSeconds": round(age, 2) if age is not None else None,
            "frameReady": True,
            "error": None,
        })
        with latest_camera_lock:
            latest_camera_result = result
        if result["name"] != "Unknown":
            print(f"[RTSP AI] {result['name']} ({result['sim']:.2f}) level={result['level']} stable={result['stable']}")
        time.sleep(max(args.interval, 0.2))


rtsp_capture_thread = None
rtsp_recognition_thread = None
if args.rtsp:
    rtsp_capture_thread = threading.Thread(target=rtsp_worker, daemon=True)
    rtsp_recognition_thread = threading.Thread(target=rtsp_recognition_worker, daemon=True)
    rtsp_capture_thread.start()
    rtsp_recognition_thread.start()


@app.route("/frame", methods=["POST"])
def receive_frame():
    npimg = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "decode_failed"}), 400

    # Browser frames are isolated from RTSP stability and never unlock physical
    # hardware. Only the configured IP-camera path may request an unlock.
    result = recognize_frame(frame, browser_stable, allow_unlock=False)
    result.update({"source": "browser", "updatedAt": time.time()})
    print(f"[AI] {result['name']} ({result['sim']:.2f}) level={result['level']} stable={result['stable']}")
    return jsonify(result)


@app.route("/camera/latest", methods=["GET"])
def camera_latest():
    with latest_camera_lock:
        payload = dict(latest_camera_result)
    with latest_frame_lock:
        updated_at = latest_frame_updated_at

    age = frame_age_seconds(updated_at)
    payload["frameUpdatedAt"] = updated_at
    payload["frameAgeSeconds"] = round(age, 2) if age is not None else None
    payload["frameReady"] = not frame_is_stale(updated_at)
    if args.rtsp and not payload["frameReady"]:
        payload.update(camera_base_result(error=payload.get("error") or "camera_frame_stale", updated_at=updated_at))
    return jsonify(payload)


@app.route("/camera/frame", methods=["GET"])
def camera_frame():
    with latest_frame_lock:
        frame = latest_frame_jpeg
        updated_at = latest_frame_updated_at

    if not frame:
        return jsonify({"error": "frame_not_ready"}), 404
    if frame_is_stale(updated_at):
        age = frame_age_seconds(updated_at)
        return jsonify({
            "error": "camera_frame_stale",
            "frameUpdatedAt": updated_at,
            "frameAgeSeconds": round(age, 2) if age is not None else None,
        }), 409

    return Response(
        frame,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "X-Frame-Updated-At": str(updated_at or ""),
        },
    )


@app.route("/health", methods=["GET"])
def health():
    with latest_frame_lock:
        frame_ready = latest_frame_jpeg is not None and not frame_is_stale(latest_frame_updated_at)

    capture_alive = bool(rtsp_capture_thread and rtsp_capture_thread.is_alive())
    recognition_alive = bool(rtsp_recognition_thread and rtsp_recognition_thread.is_alive())
    workers_healthy = not args.rtsp or (capture_alive and recognition_alive)
    payload = {
        "status": "ok" if workers_healthy else "degraded",
        "model": "buffalo_l",
        "rtsp": bool(args.rtsp),
        "captureWorkerAlive": capture_alive,
        "recognitionWorkerAlive": recognition_alive,
        "frameReady": frame_ready,
        "previewFps": args.preview_fps,
        "recognitionInterval": args.interval,
        "frameStaleSeconds": args.frame_stale_seconds,
        "provider": active_providers[0],
        "availableProviders": onnx_providers,
        "modelSessionProviders": model_session_providers,
        "detSize": active_det_size,
        "opencvThreads": cv2.getNumThreads(),
        "unlockEnabled": args.unlock_enabled,
        "zone": args.zone or None,
        "zoneSource": "configured" if args.zone else None,
        "marker": args.marker or None,
        "markerSource": "configured" if args.marker else None,
        "objectPlacementStatus": args.object_placement or None,
        "objectPlacementSource": "configured" if args.object_placement else None,
    }
    return jsonify(payload), 200 if workers_healthy else 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False, use_reloader=False)
