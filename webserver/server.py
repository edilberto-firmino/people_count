import os
import threading
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = os.environ.get("YOLO_MODEL", os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt"))
SOURCE = os.environ.get("YOLO_SOURCE", "0")  # "0" for default camera or path/RTSP URL
BACKEND = os.environ.get("YOLO_BACKEND", "auto")  # auto|msmf|dshow (Windows)
REQ_WIDTH = int(os.environ.get("YOLO_WIDTH", "0"))
REQ_HEIGHT = int(os.environ.get("YOLO_HEIGHT", "0"))
CONF_TH = float(os.environ.get("YOLO_CONF", "0.25"))
DIR_POLARITY = int(os.environ.get("YOLO_DIR_POLARITY", "1"))  # 1 or -1 to flip IN/OUT mapping

# -----------------------------
# Inference Worker
# -----------------------------
class InferenceState:
    def __init__(self) -> None:
        self.model = YOLO(MODEL_PATH)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated: Optional[np.ndarray] = None
        self.latest_count: int = 0
        self.fps: float = 0.0
        self.running = False
        # Tracking and line-crossing
        self.track_next_id: int = 1
        # id -> { 'centroid': (x,y), 'last_side': int, 'counted': bool, 'last_seen': float }
        self.tracks: Dict[int, Dict] = {}
        self.in_count: int = 0
        self.out_count: int = 0
        # Fixed virtual line in normalized coords (0..1). Horizontal middle by default.
        # p1 ---- p2 across frame width at mid-height
        self.line_p1_norm: Tuple[float, float] = (0.1, 0.5)
        self.line_p2_norm: Tuple[float, float] = (0.9, 0.5)
        # Tuning
        self.match_max_dist: float = 80.0  # pixels
        self.track_ttl: float = 1.0  # seconds to keep lost tracks

    def _parse_source(self, s):
        try:
            return int(s)
        except Exception:
            return s

    def _backend_flag(self, name: str):
        if name == "msmf":
            return cv2.CAP_MSMF
        if name == "dshow":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def _try_open(self, source, backend_name):
        flag = self._backend_flag(backend_name)
        cap = cv2.VideoCapture(source, flag)
        if REQ_WIDTH > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
        if REQ_HEIGHT > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _point_side(self, a: Tuple[int, int], b: Tuple[int, int], p: Tuple[int, int]) -> int:
        """Return which side of the line AB the point P lies on.
        >0 means one side, <0 the other, 0 means colinear. We normalize to -1, 0, 1.
        """
        ax, ay = a
        bx, by = b
        px, py = p
        # Cross product (B - A) x (P - A)
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        else:
            return 0

    def ensure_capture(self):
        if self.cap is not None and self.cap.isOpened():
            return
        src = self._parse_source(SOURCE)
        order = ["msmf", "dshow", "auto"] if BACKEND == "auto" else [BACKEND, "dshow" if BACKEND == "msmf" else "msmf", "auto"]
        for be in order:
            cap = self._try_open(src, be if be != "auto" else "auto")
            if cap is not None:
                self.cap = cap
                return
        raise RuntimeError(f"Unable to open video source: {SOURCE} (tried backends: {order})")

    def start(self):
        if self.running:
            return
        self.ensure_capture()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_time = time.time()
        while self.running:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok or frame is None:
                # Try to reopen after short delay
                time.sleep(0.5)
                try:
                    self.ensure_capture()
                except Exception:
                    pass
                continue

            start_t = time.time()
            results = self.model(frame, conf=CONF_TH, verbose=False)
            count = 0
            annotated = frame.copy()

            # Build detections list (centroids) for persons
            detections: List[Tuple[int, int, int, int, Tuple[int,int]]] = []  # (x1,y1,x2,y2,(cx,cy))
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls)
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    detections.append((int(x1), int(y1), int(x2), int(y2), (cx, cy)))
                count = len(detections)

            # Update tracking and line crossing
            h, w = frame.shape[:2]
            p1 = (int(self.line_p1_norm[0] * w), int(self.line_p1_norm[1] * h))
            p2 = (int(self.line_p2_norm[0] * w), int(self.line_p2_norm[1] * h))

            now = time.time()
            # Mark tracks as stale if not seen recently
            to_delete = []
            for tid, tr in self.tracks.items():
                if now - tr.get('last_seen', 0.0) > self.track_ttl:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]

            # Match detections to existing tracks by nearest centroid
            unmatched_dets = list(range(len(detections)))
            # Build list of current track ids
            track_ids = list(self.tracks.keys())
            used_tracks = set()
            for di in list(unmatched_dets):
                _, _, _, _, (cx, cy) = detections[di]
                # find nearest track
                best_tid = None
                best_dist = 1e9
                for tid in track_ids:
                    if tid in used_tracks:
                        continue
                    tcx, tcy = self.tracks[tid]['centroid']
                    dist = ((tcx - cx) ** 2 + (tcy - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None and best_dist <= self.match_max_dist:
                    # update track
                    self.tracks[best_tid]['centroid'] = (cx, cy)
                    self.tracks[best_tid]['last_seen'] = now
                    used_tracks.add(best_tid)
                    unmatched_dets.remove(di)
                # else leave unmatched

            # Create new tracks for unmatched detections
            for di in unmatched_dets:
                _, _, _, _, (cx, cy) = detections[di]
                side = self._point_side(p1, p2, (cx, cy))
                self.tracks[self.track_next_id] = {
                    'centroid': (cx, cy),
                    'last_side': side,
                    'counted': False,
                    'last_seen': now,
                }
                used_tracks.add(self.track_next_id)
                self.track_next_id += 1

            # Check for line crossing events on updated tracks
            for tid in list(used_tracks):
                tr = self.tracks.get(tid)
                if tr is None:
                    continue
                cx, cy = tr['centroid']
                current_side = self._point_side(p1, p2, (cx, cy))
                last_side = tr.get('last_side', current_side)
                # If precisely on the line, don't change side yet
                side_for_logic = current_side if current_side != 0 else last_side
                if side_for_logic != 0 and last_side != 0 and side_for_logic != last_side and not tr.get('counted', False):
                    direction = 1 if last_side < side_for_logic else -1
                    mapped = direction * DIR_POLARITY
                    if mapped > 0:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    tr['counted'] = True
                tr['last_side'] = side_for_logic

            # Draw overlays
            annotated = frame.copy()
            # Draw line
            cv2.line(annotated, p1, p2, (0, 255, 255), 2)
            # Draw detections and track ids
            for tid, tr in self.tracks.items():
                cx, cy = tr['centroid']
                color = (0, 255, 0) if not tr.get('counted', False) else (200, 200, 200)
                cv2.circle(annotated, (int(cx), int(cy)), 5, color, -1)
                cv2.putText(annotated, f"ID:{tid}", (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Header text
            cv2.putText(annotated, f"IN: {self.in_count}  OUT: {self.out_count}  Total: {self.in_count + self.out_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            end_t = time.time()
            cur_fps = 1.0 / max(1e-6, (end_t - start_t))

            with self.frame_lock:
                self.latest_frame = frame
                self.latest_annotated = annotated
                self.latest_count = count
                self.fps = cur_fps

        if self.cap is not None:
            self.cap.release()
            self.cap = None


state = InferenceState()

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="People Counting WebServer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class CountResponse(BaseModel):
    count: int
    fps: float


class ConfigRequest(BaseModel):
    conf: float


@app.on_event("startup")
async def _on_startup():
    state.start()


@app.get("/")
async def index():
    # Serve a minimal index that references the static file
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/html")
    return Response("<h1>People Counting</h1><p>Static index.html not found.</p>", media_type="text/html")


@app.get("/count", response_model=CountResponse)
async def get_count():
    with state.frame_lock:
        return CountResponse(count=state.latest_count, fps=state.fps)


@app.get("/counts")
async def get_counts():
    with state.frame_lock:
        return {
            "in": state.in_count,
            "out": state.out_count,
            "total": state.in_count + state.out_count,
        }


@app.get("/snapshot")
async def snapshot():
    with state.frame_lock:
        frame = state.latest_annotated if state.latest_annotated is not None else state.latest_frame
        if frame is None:
            return Response(status_code=503)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return Response(status_code=500)
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


@app.post("/config")
async def set_config(cfg: ConfigRequest):
    global CONF_TH
    # Clamp value between 0 and 1
    CONF_TH = float(max(0.0, min(1.0, cfg.conf)))
    return {"conf": CONF_TH}


@app.get("/health")
async def health():
    with state.frame_lock:
        has_frame = state.latest_frame is not None
        cur_conf = CONF_TH
    return {"status": "ok", "has_frame": has_frame, "conf": cur_conf}


@app.get("/video")
async def video_stream():
    def generate():
        boundary = "frame"
        while True:
            with state.frame_lock:
                frame = state.latest_annotated if state.latest_annotated is not None else state.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)

    return Response(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# For local execution: uvicorn webserver.server:app --reload --host 0.0.0.0 --port 8000

    
    
    
    
