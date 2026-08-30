import argparse
import csv
import queue
import re
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pytesseract
from ultralytics import YOLO

try:
    from PIL import Image
    from tesserocr import PSM, PyTessBaseAPI
    TESSEROCR_AVAILABLE = True
except ImportError:
    TESSEROCR_AVAILABLE = False
    
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "license_plate_detector.pt"
alloweds_path = BASE_DIR / "allowed_plates.txt"

header = ["Fecha y hora", "id vehiculo", "Matricula", "Resultado de acceso", "Lecturas OCR", "Fotograma", "Fuente", "Deteccion de YOLO (ms)", "OCR (ms)", "Tiempo hasta confirmacion (s)", "Consumo CPU"]

@dataclass
class PlateTrack:
    box: tuple[int, int, int, int]
    last_seen: int
    last_seen_at: float = field(default_factory=time.monotonic)
    last_ocr_at: float = 0.0
    pending: bool = False
    votes: Counter = field(default_factory=Counter)
    readings: list[str] = field(default_factory=list)
    text: str = ""
    saved: bool = False
    authorized: bool | None = None
    first_seen_at: float = field(default_factory=time.monotonic)
    cpu_start: float = field(default_factory=time.process_time)
    detection_time_ms: float = 0.0
    ocr_time_ms: float = 0.0

### LED ---
class AccessLed:
    RED, GREEN, BLUE = 17, 27, 22  # Pines físicos 11, 13 y 15

    def __init__(self, common_anode: bool = False) -> None:
        try:
            import RPi.GPIO as gpio
        except ImportError as exc:
            raise RuntimeError("RPi.GPIO no está instalado; el LED solo puede usarse en una Raspberry Pi.") from exc
        self.gpio = gpio
        self.common_anode = common_anode
        self.reset_at = 0.0
        gpio.setwarnings(False)
        gpio.setmode(gpio.BCM)
        for pin in (self.RED, self.GREEN, self.BLUE):
            gpio.setup(pin, gpio.OUT, initial=self._level(False))
        self.set_color(0, 0, 1)  # AZUL: sistema preparado

    def _level(self, on: bool) -> int:
        return int(not on) if self.common_anode else int(on)

    def set_color(self, red: int, green: int, blue: int) -> None:
        for pin, value in zip((self.RED, self.GREEN, self.BLUE), (red, green, blue)):
            self.gpio.output(pin, self._level(bool(value)))

    def signal_access(self, authorized: bool, seconds: float = 2.0) -> None:
        self.set_color(0, 1, 0) if authorized else self.set_color(1, 0, 0)
        self.reset_at = time.monotonic() + seconds

    def tick(self) -> None:
        if self.reset_at and time.monotonic() >= self.reset_at:
            self.set_color(0, 0, 1)
            self.reset_at = 0.0

    def close(self) -> None:
        self.set_color(0, 0, 0)
        self.gpio.cleanup((self.RED, self.GREEN, self.BLUE))
### LED ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconocimiento de matriculas para Raspberry Pi")
    parser.add_argument("--source", default="camera", help="'camera' o ruta de vídeo")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Seleccione la region de interes de la imagen; reduce mucho el uso de la CPU")
    parser.add_argument("--imgsz", type=int, default=416, help="Resolución de inferencia YOLO (416 por defecto)")
    parser.add_argument("--detecter", type=int, default=2, help="Detectar una vez cada N fotogramas (2 por defecto)")
    parser.add_argument("--t-ocr", type=float, default=0.8, help="Espera al menos (0,8 s) antes de volver a intentar reconocer su matrícula")
    parser.add_argument("--confirmations", type=int, default=2, help="Lecturas iguales necesarias para guardar (2 por defecto)")
    parser.add_argument("--confidence", type=float, default=0.45, help="Confianza mínima del detector de matrículas (0.45 por defecto)")
    parser.add_argument("--last-frame", default="last_frame.jpg", help="Imagen de estado (vacío para desactivarla)")
    parser.add_argument("--show", action="store_true", help="Mostrar ventana (no usar por SSH sin escritorio)")
    parser.add_argument("--led", action="store_true", help="Activar LED RGB de acceso, tenemos 'allowed_plates.txt' que serían las matriculas permitidas en la carpeta")
    parser.add_argument("--debug", action="store_true", help="Modo depuración")

    return parser.parse_args()


def normalize_plate(text: str) -> str:
    """Devuelve una matrícula válida (ABC1234 o 1234ABC) o una cadena vacía"""
    raw = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = re.search(r"(?:\d{4}[A-Z]{3}|[A-Z]{3}\d{4})", raw)
    if match:
        return match.group(0)
    if len(raw) != 7:
        return ""
    
    # si match: corrige errores típicos de OCR
    digit_map = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
    letter_map = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})
    
    candidate = raw[:4].translate(digit_map) + raw[4:].translate(letter_map)
    if re.fullmatch(r"\d{4}[A-Z]{3}", candidate):
        return candidate

    candidate2 = raw[:3].translate(letter_map) + raw[3:].translate(digit_map)
    return candidate2 if re.fullmatch(r"[A-Z]{3}\d{4}", candidate2) else ""

def plate_crop(frame, box: tuple[int, int, int, int]):
    """Recorta la imagen según la caja detectada, asegurando que no se salga de los límites"""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    return frame[max(0, y1):min(height, y2),max(0, x1):min(width, x2)]

def read_plate(image, ocr = None, debug: bool = False) -> str:
    """OCR de una matrícula recortada"""
     # Se ejecuta en el hilo de OCR
    
    if image is None or image.size == 0:
        return ""

    # preprocesamiento de la imagen para mejorar el OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    
    candidates = []
    raw_reads = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if debug:
        cv2.imwrite(str(BASE_DIR/"ocr_otsu.png"), otsu)

    if ocr:
        ocr.SetImage(Image.fromarray(otsu))
        raw_text = ocr.GetUTF8Text() or ""
    else:
        config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw_text = pytesseract.image_to_string(otsu, config=config)
        
    raw_reads.append(raw_text.strip())
    candidate = normalize_plate(raw_text)
    if candidate:
        candidates.append(candidate)
    result = Counter(candidates).most_common(1)[0][0] if candidates else ""
    if debug:
        print(f"OCR bruto: {raw_reads} | valido {result or 'ninguno'}")
    return result


def iou(caja1, caja2):
    """Calcula el IoU entre dos cajas """
    x1, y1, x2, y2 = caja1
    x3, y3, x4, y4 = caja2

    xx1 = max(x1, x3)
    yy1 = max(y1, y3)
    xx2 = min(x2, x4)
    yy2 = min(y2, y4)

    area_interseccion = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    area_total = ((x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - area_interseccion)

    return area_interseccion / area_total if area_total else 0 # 0 - no se solapan, 1 - cajas idénticas

def comparar_cajas(caja_actual, caja_anterior):
    solapamiento = iou(caja_actual, caja_anterior)

    centro_x1 = (caja_actual[0] + caja_actual[2]) / 2
    centro_y1 = (caja_actual[1] + caja_actual[3]) / 2
    centro_x2 = (caja_anterior[0] + caja_anterior[2]) / 2
    centro_y2 = (caja_anterior[1] + caja_anterior[3]) / 2

    distancia = ((centro_x1 - centro_x2) ** 2 +
                 (centro_y1 - centro_y2) ** 2) ** 0.5

    ancho = max(caja_actual[2] - caja_actual[0],
                caja_anterior[2] - caja_anterior[0])

    cercania = max(0, 1 - distancia / max(35, ancho * 1.25))

    return max(solapamiento, cercania)

def validate_box(box, width, height):
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 - x1 < 60 or y2 - y1 < 18:
        return None
    return x1, y1, x2, y2


def open_camera():
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("Picamera2 no está instalado. Usa --source vídeo o instala python3-picamera2.") from exc
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"size": (640,480), "format": "RGB888"}))
    camera.start()
    time.sleep(1.5)
    return camera

def load_allowed_plates(path=alloweds_path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No existe la lista de autorizados: {file_path}")
    return {
        plate
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
        for plate in (normalize_plate(line),)
        if plate
    }

def open_detecter():
    # alamacena los resultados en un CSV
    csv_path = BASE_DIR / "detected_plates.csv"
    new_file = not csv_path.exists()
    file = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(file)
    if new_file:
        writer.writerow(header)
        file.flush()
    return file, writer

def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No encuentro el modelo de matrículas: {MODEL_PATH}")
    if args.detecter < 1 or args.confirmations < 1:
        raise ValueError("--detecter y --confirmations deben ser mayores que cero")
    led = None
    allowed_plates: set[str] = set()
    if args.led:
        allowed_plates = load_allowed_plates()
        led = AccessLed()
        print(f"* * ----- Control de acceso activo: {len(allowed_plates)} matrículas autorizadas.")
    model = YOLO(str(MODEL_PATH))
    csv_file, writer = open_detecter()

    jobs: queue.Queue[tuple[int, object] | None] = queue.Queue(maxsize=1)
    results: queue.Queue[tuple[int, str]] = queue.Queue()
    yolo_jobs: queue.Queue[tuple[object, int, int, int, int] | None] = queue.Queue(maxsize=1)
    yolo_results: queue.Queue[tuple[list, float]] = queue.Queue(maxsize=1)
    
    def ocr_worker() -> None: # HILO 1: LEE MATRICULAS PARA QUE NO BLOQUEE EL HILO PRINCIPAL
        ocr_api = None
        if TESSEROCR_AVAILABLE:
            try:
                ocr_api = PyTessBaseAPI(path="/usr/share/tesseract-ocr/5/tessdata",lang="eng",psm=PSM.SINGLE_LINE)
                ocr_api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                print("## OCR persistente activado (tesserocr)")
            except RuntimeError as exc:
                print(f"No se pudo iniciar tesserocr: {exc}. Se usará pytesseract.")
        while True:
            job = jobs.get()
            if job is None:
                if ocr_api:
                    ocr_api.End()
                return
            track_id, crop = job
            try:
                start_ocr = time.perf_counter()
                text = read_plate(crop, ocr_api, args.debug)
                results.put((track_id, text, (time.perf_counter() - start_ocr) * 1000))
            except Exception as exc:  # el vídeo debe seguir aunque falle una lectura
                print(f"OCR falló para ID {track_id}: {exc}")
                results.put((track_id, "",0.0))
            finally:
                jobs.task_done()

    worker = threading.Thread(target=ocr_worker, daemon=True)
    worker.start()
    # -- HILO 1

    def yolo_worker() -> None: # HILO 2: DETECTA MATRICULAS PARA QUE NO BLOQUEE EL HILO PRINCIPAL
        while True:
            job = yolo_jobs.get()
            if job is None:
                return

            roi, x, y, rw, rh = job
            try:
                start_detection = time.perf_counter()
                detections = model(roi, imgsz=args.imgsz, conf=args.confidence, verbose=False)[0]
                detection_time_ms = (time.perf_counter() - start_detection) * 1000

                boxes = []
                for data in detections.boxes.xyxy.cpu().tolist():
                    box = validate_box(data, rw, rh)
                    if box:
                        boxes.append((box[0] + x, box[1] + y, box[2] + x, box[3] + y))

                try:
                    yolo_results.put_nowait((boxes, detection_time_ms))
                except queue.Full:
                    pass
            finally:
                yolo_jobs.task_done()

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()
    # -- HILO 2 
    
    camera = None
    capture = None
    if args.source == "camera":
        camera = open_camera()
    else:
        capture = cv2.VideoCapture(args.source)
        if not capture.isOpened():
            raise RuntimeError(f"No se puede abrir la fuente: {args.source}")

    tracks: dict[int, PlateTrack] = {}
    next_track_id = 1
    frame_number = 0
    roi_selection = args.roi

    def update_tracks(boxes, detection_time_ms):
        nonlocal next_track_id
        unmatched = set(tracks)

        for box in boxes:
            matches = [
                (comparar_cajas(box, tracks[track_id].box), track_id)
                for track_id in unmatched
            ]
            score, track_id = max(matches, default=(0.0, -1))

            if score < 0.35:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = PlateTrack(box=box, last_seen=frame_number, detection_time_ms=detection_time_ms)
            else:
                unmatched.discard(track_id)
                track = tracks[track_id]
                track.box = box
                track.last_seen = frame_number
                track.last_seen_at = time.monotonic()
                track.detection_time_ms = detection_time_ms

    print("Sistema iniciado. Pulsa q para salir si usas --show.")

    try:
        while True: # HILO PRINCIPAL: CAPTURA, DIBUJA Y GESTIONA LOS HILOS DE YOLO Y OCR
            if camera:
                frame = camera.capture_array()
            else:
                frame = capture.read()[1]
            if frame is None:
                break
            clean_frame = frame.copy()
            frame_number += 1
            height, width = frame.shape[:2]

            if roi_selection:
                x, y, rw, rh = roi_selection
                x, y = max(0, x), max(0, y)
                rw, rh = min(rw, width - x), min(rh, height - y)
            else:
                x, y, rw, rh = 0, 0, width, height
            roi = frame[y:y + rh, x:x + rw]
            if roi.size == 0:
                raise ValueError("La ROI queda fuera de la imagen")
            
            if camera:
                # Cámara: se muestra el vídeo mientras YOLO procesa el último fotograma
                if frame_number % args.detecter == 0:
                    try:
                        yolo_jobs.put_nowait((roi.copy(), x, y, rw, rh))
                    except queue.Full:
                        pass

                while True:
                    try:
                        boxes, detection_time_ms = yolo_results.get_nowait()
                    except queue.Empty:
                        break
                    update_tracks(boxes, detection_time_ms)
            elif frame_number % args.detecter == 0:
                # Vídeo de ejemplo: se procesa de forma síncrona para no saltar fotogramas
                start_detection = time.perf_counter()
                detections = model(roi, imgsz=args.imgsz, conf=args.confidence, verbose=False)[0]
                detection_time_ms = (time.perf_counter() - start_detection) * 1000

                boxes = []
                for data in detections.boxes.xyxy.cpu().tolist():
                    box = validate_box(data, rw, rh)
                    if box:
                        boxes.append((box[0] + x, box[1] + y, box[2] + x, box[3] + y))
                update_tracks(boxes, detection_time_ms)

            tracks = {
                track_id: track
                for track_id, track in tracks.items()
                if track.pending or time.monotonic() - track.last_seen_at <= 2.0
            }

            # Recoger el OCR terminado sin bloquear la captura ni YOLO.
            while True:
                try:
                    track_id, text, ocr_time_ms = results.get_nowait()
                except queue.Empty:
                    break
                track = tracks.get(track_id)
                if track:
                    track.pending = False
                    track.ocr_time_ms = ocr_time_ms
                    if text:
                        track.votes[text] += 1
                        track.readings.append(text)
                        best, votes = track.votes.most_common(1)[0]
                        confirmed = best if votes >= args.confirmations else ""
                        
                        if confirmed:
                            track.text = confirmed
                            track.authorized = confirmed in allowed_plates if led else None
                            if not track.saved:
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                if led:
                                    access_result = "permitido" if track.authorized else "denegado"
                                else:
                                    access_result = "sin_comprobar"
                                elapsed = time.monotonic() - track.first_seen_at
                                cpu_average = 100 * (time.process_time() - track.cpu_start) / max(elapsed, 0.001)
                                writer.writerow([
                                    timestamp, track_id, confirmed, access_result,
                                    len(track.readings), frame_number, args.source,
                                    round(track.detection_time_ms, 1), round(track.ocr_time_ms, 1),
                                    round(elapsed, 2), round(cpu_average, 1)
                                ])
                                csv_file.flush()
                                track.saved = True
                                if led:
                                    led.signal_access(track.authorized)
                                    result = "ACCESO PERMITIDO" if track.authorized else "ACCESO DENEGADO"
                                    print(f"{result}: {confirmed} (ID {track_id})")
                                else:
                                    print(f"MATRÍCULA CONFIRMADA: {confirmed} (ID {track_id})")

            now = time.monotonic()
            if led:
                led.tick()
            for track_id, track in tracks.items():
                x1, y1, x2, y2 = track.box
                color = (0, 200, 0) if track.text and track.authorized is not False else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if track.text and led:
                    label = f"{track.text} {'OK' if track.authorized else 'DENEGADO'}"
                else:
                    label = track.text or f"leyendo ID {track_id}"
                    
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # No repetir OCR de una matrícula confirmada ni encolar si el hilo está ocupado.
                if track.text or track.pending or now - track.last_ocr_at < args.t_ocr:
                    continue
                crop = plate_crop(clean_frame, track.box)
                if crop.size and cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() > 8:
                    if args.debug and args.last_frame:
                        cv2.imwrite(str(BASE_DIR / args.last_frame), frame)
                    if args.show and args.debug:
                        enlarged_crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Imagen enviada al OCR", enlarged_crop)
                    try:
                        jobs.put_nowait((track_id, crop.copy()))
                        track.pending = True
                        track.last_ocr_at = now
                    except queue.Full:
                        pass

            if roi_selection:
                cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 0), 2)
                
            if args.show:
                cv2.imshow("RoadWatch", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    selection = cv2.selectROI("Selecciona ROI y pulsa Enter", clean_frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Selecciona ROI y pulsa Enter")
                    if selection[2] and selection[3]:
                        roi_selection = tuple(int(value) for value in selection)
                        print(f"ROI seleccionada: {roi_selection}")
                        
    finally:
        try:
            yolo_jobs.put_nowait(None)
        except queue.Full:
            pass

        yolo_thread.join(timeout=2)
        csv_file.close()
        jobs.put(None)
        worker.join(timeout=3)
        if camera:
            camera.stop()
        if capture:
            capture.release()
        if led:
            led.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()