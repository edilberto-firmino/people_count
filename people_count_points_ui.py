import cv2
from ultralytics import YOLO
import time
import os
import argparse
import numpy as np

# =============================
# Configurações iniciais
# =============================
# Caminho do modelo. Para contar galinhas, use um modelo YOLO treinado para aves/galinhas (ex: 'best.pt').
# Padrão: yolov8n.pt (COCO) — NÃO tem classe "chicken" por padrão.
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# Fonte de vídeo padrão: 0 = webcam. Para arquivo, altere para caminho do vídeo.
DEFAULT_SOURCE = os.environ.get("YOLO_SOURCE", "0")  # "0" (webcam) ou caminho de vídeo
DEFAULT_BACKEND = os.environ.get("YOLO_BACKEND", "auto")  # auto|msmf|dshow
DEFAULT_WIDTH = int(os.environ.get("YOLO_WIDTH", "0"))
DEFAULT_HEIGHT = int(os.environ.get("YOLO_HEIGHT", "0"))

# =============================
# Estado global e helpers
# =============================
class AppState:
    def __init__(self, source, backend, width, height):
        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.source = source
        self.backend = backend
        self.req_width = width
        self.req_height = height
        self.window_name = "Contagem - Pontos (UI)"
        self.trackbar_win = "Controles"
        self.conf_th = 25  # 0-100 (representa 0.0-1.0)
        self.point_size = 6  # raio do ponto desenhado
        self.selected_class_id = -1  # -1 = todas
        self.pause = False
        self.show_help = True
        self.roi_points = []  # lista de pontos (x, y)
        self.drawing_roi = False
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _parse_source(self, s):
        # converte "0" -> 0, mantém string para arquivo/rtsp/rtmp
        try:
            return int(s)
        except (TypeError, ValueError):
            return s

    def _backend_flag(self, name):
        if name == "msmf":
            return cv2.CAP_MSMF
        if name == "dshow":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def _try_open(self, source, backend_name):
        flag = self._backend_flag(backend_name)
        cap = cv2.VideoCapture(source, flag)
        if self.req_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        if self.req_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def ensure_capture(self):
        if self.cap is not None and self.cap.isOpened():
            return

        src = self._parse_source(self.source)
        tried = []

        order = []
        if self.backend in ("msmf", "dshow"):
            order = [self.backend, "dshow" if self.backend == "msmf" else "msmf", "auto"]
        else:
            order = ["msmf", "dshow", "auto"]

        for be in order:
            tried.append(be)
            cap = self._try_open(src, be if be != "auto" else "auto")
            if cap is not None:
                self.cap = cap
                return

        msg = [
            f"Não foi possível abrir a fonte de vídeo: {self.source}",
            f"Backends testados: {tried}",
            "Dicas:",
            " - Feche outros programas que possam estar usando a câmera (Teams/Zoom/Browser)",
            " - Tente outro índice de câmera: --source 1 ou 2",
            " - Forçar backend: --backend dshow",
            " - Usar arquivo de vídeo: --source caminho\\video.mp4",
        ]
        raise RuntimeError("\n".join(msg))

state = None

# =============================
# Utilitários
# =============================

def draw_help(frame):
    if not state.show_help:
        return
    h, w = frame.shape[:2]
    lines = [
        "Teclas:",
        " q: sair",
        " h: mostrar/ocultar ajuda",
        " p: pausar/continuar",
        " c: limpar ROI",
        " r: alternar modo desenho ROI (clique para adicionar pontos; ENTER para fechar)",
        " m: alternar classe a contar (entre classes detectadas)",
        " + / -: ajustar tamanho do ponto",
        " setaEsq/Dir: reduzir/aumentar conf",
        "Clique & ENTER: fechar polígono ROI",
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def inside_roi(pt):
    if len(state.roi_points) < 3:
        return True  # sem ROI definida => conta tudo
    contour = cv2.convexHull(np.array(state.roi_points, dtype=np.int32)) if len(state.roi_points) >= 3 else None
    if contour is None:
        return True
    # pointPolygonTest retorna >0 dentro, 0 na borda, <0 fora
    res = cv2.pointPolygonTest(contour, (int(pt[0]), int(pt[1])), False)
    return res >= 0


def draw_roi(frame):
    if len(state.roi_points) >= 1:
        for i, p in enumerate(state.roi_points):
            cv2.circle(frame, p, 3, (0, 255, 255), -1)
            if i > 0:
                cv2.line(frame, state.roi_points[i - 1], p, (0, 255, 255), 1)
    if len(state.roi_points) >= 3:
        # fecha o polígono ligando último ao primeiro (visual)
        cv2.line(frame, state.roi_points[-1], state.roi_points[0], (0, 255, 255), 1)


def on_trackbar_conf(val):
    state.conf_th = val


def on_trackbar_point_size(val):
    state.point_size = max(1, val)


def on_mouse(event, x, y, flags, param):
    if state.drawing_roi:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.roi_points.append((x, y))


def build_controls():
    cv2.namedWindow(state.trackbar_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(state.trackbar_win, 400, 120)
    cv2.createTrackbar("Conf(%)", state.trackbar_win, state.conf_th, 100, on_trackbar_conf)
    cv2.createTrackbar("Ponto", state.trackbar_win, state.point_size, 20, on_trackbar_point_size)


def cycle_class_id(detected_class_ids):
    # detected_class_ids: lista ordenada de IDs presentes no frame
    if not detected_class_ids:
        return
    if state.selected_class_id == -1:
        state.selected_class_id = detected_class_ids[0]
    else:
        try:
            idx = detected_class_ids.index(state.selected_class_id)
            idx = (idx + 1) % len(detected_class_ids)
            state.selected_class_id = detected_class_ids[idx]
        except ValueError:
            state.selected_class_id = detected_class_ids[0]


def class_name(cls_id):
    names = getattr(state.model, "names", None)
    if isinstance(names, dict):
        return names.get(int(cls_id), str(int(cls_id)))
    if isinstance(names, list) and 0 <= int(cls_id) < len(names):
        return names[int(cls_id)]
    return str(int(cls_id))


# =============================
# Loop principal
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="Contagem com pontos (UI)")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Fonte: índice da webcam (ex. 0) ou caminho do vídeo")
    p.add_argument("--backend", type=str, default=DEFAULT_BACKEND, choices=["auto", "msmf", "dshow"], help="Forçar backend de captura (Windows)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Largura desejada da câmera")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Altura desejada da câmera")
    return p.parse_args()


def main():
    args = parse_args()
    print("Carregando modelo:", MODEL_PATH)

    global state
    state = AppState(source=args.source, backend=args.backend, width=args.width, height=args.height)
    state.ensure_capture()

    cv2.namedWindow(state.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(state.window_name, on_mouse)
    build_controls()

    while True:
        if not state.pause:
            ok, frame = state.cap.read()
            if not ok:
                print("Fim do vídeo ou erro ao ler frame.")
                break
        else:
            # em pausa: não lê novo frame, só reexibe
            ok, frame = True, frame

        start_t = time.time()

        # Detecção com threshold de confiança
        conf_val = max(0.0, min(1.0, state.conf_th / 100.0))
        results = state.model(frame, conf=conf_val, verbose=False)

        people_count = 0
        total_count = 0
        detected_class_ids = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls)
                score = float(box.conf) if hasattr(box, 'conf') else 0.0
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if cls not in detected_class_ids:
                    detected_class_ids.append(cls)

                # Filtro de classe (se selecionada)
                if state.selected_class_id != -1 and cls != state.selected_class_id:
                    continue

                # ROI
                if not inside_roi((cx, cy)):
                    continue

                total_count += 1
                # Desenho do ponto
                cv2.circle(frame, (cx, cy), state.point_size, (255, 0, 255), -1)
                # Opcional: desenhar bbox leve
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (120, 50, 200), 1)

        # FPS
        end_t = time.time()
        dt = end_t - state.last_fps_time
        if dt > 0:
            state.fps = 1.0 / (end_t - start_t + 1e-6)
        state.last_fps_time = end_t

        # Desenhar ROI e overlays
        draw_roi(frame)
        draw_help(frame)

        # Headers
        conf_str = f"conf: {int(conf_val*100)}%"
        cls_str = "todas" if state.selected_class_id == -1 else f"{class_name(state.selected_class_id)} ({state.selected_class_id})"
        cv2.putText(frame, f"Contagem: {total_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Classe: {cls_str}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 0), 2)
        cv2.putText(frame, f"FPS: {state.fps:.1f} | {conf_str}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        cv2.imshow(state.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            state.show_help = not state.show_help
        elif key == ord('p'):
            state.pause = not state.pause
        elif key == ord('c'):
            state.roi_points = []
        elif key == ord('r'):
            state.drawing_roi = not state.drawing_roi
        elif key == 13:  # ENTER: finalizar polígono (apenas visual)
            state.drawing_roi = False
        elif key == ord('m'):
            detected_class_ids.sort()
            cycle_class_id(detected_class_ids)
        elif key == ord('+') or key == ord('='):
            state.point_size = min(50, state.point_size + 1)
            cv2.setTrackbarPos("Ponto", state.trackbar_win, state.point_size)
        elif key == ord('-') or key == ord('_'):
            state.point_size = max(1, state.point_size - 1)
            cv2.setTrackbarPos("Ponto", state.trackbar_win, state.point_size)
        elif key == 81:  # seta esquerda
            state.conf_th = max(0, state.conf_th - 1)
            cv2.setTrackbarPos("Conf(%)", state.trackbar_win, state.conf_th)
        elif key == 83:  # seta direita
            state.conf_th = min(100, state.conf_th + 1)
            cv2.setTrackbarPos("Conf(%)", state.trackbar_win, state.conf_th)

    if state.cap is not None:
        state.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
