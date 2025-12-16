import cv2                      # OpenCV (cv2): librería para visión artificial y procesamiento de imágenes.
from ultralytics import YOLO
import easyocr                  # EasyOCR: librería para reconocimiento óptico de caracteres (OCR).
import numpy as np              # NumPy: librería para trabajar con matrices y vectores multidimensionales.
import imutils                  # imutils: librería con funciones de utilidad para procesamiento de imágenes con OpenCV.
import rembg                    # rembg: librería para eliminar el fondo de una imagen.
from PIL import Image           # PIL (Pillow): librería para abrir, manipular y guardar imágenes.


print("Cargando modelo YOLO...")
model = YOLO("model/placas.pt") 

print("Inicializando cámara...")

cap = cv2.VideoCapture(0)  # 0 = cámara por defecto (USB o PiCam adaptada)
cap.set(640, 480)

reader = easyocr.Reader(['en'])   # OCR basado en IA

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer de la cámara.")
        break

    # 3. Detección con el modelo YOLO
    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Filtro de confianza
            if conf < 0.5:
                continue

            # Dibujar el rectángulo en la imagen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Placa {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 4. Recortar matrícula
            placa = frame[y1:y2, x1:x2]
            cv2.imwrite("ultima_placa.jpg", placa)

            # 5. Aplicar OCR IA
            ocr_result = reader.readtext(placa, detail=0)

            if len(ocr_result) > 0:
                print("Matrícula detectada:", ocr_result[0])
                cv2.putText(frame, ocr_result[0], (x1, y2+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 6. Mostrar imagen
    cv2.imshow("Reconocimiento de matrículas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
