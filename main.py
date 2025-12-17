import cv2
import numpy as np
import csv
import os
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from cvzone.Utils import putTextRect

ocr = PaddleOCR(use_textline_orientation=True, lang='en')
yolo_model = YOLO('yolo11x.pt') # modelo YOLOv11x preentrenado para detección general
license_plates_detector = YOLO("license_plate_detector.pt") # modelo YOLO preentrenado especializado en detección de matrículas

filename = 'detected_plates.csv'
if not os.path.exists(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame number', 'Car ID', 'License plate'])

vehicles = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'} # clases de vehículos que detecta el modelo
frame_number = 0
tracker = "bytetrack.yaml"
vehicle_plates = {}

""" PASO 1: REPRODUCCIÓN DEL VIDEO + SELECCIÓN DE LA REGIÓN DE INTERÉS (ROI) """
ejemplo = cv2.VideoCapture('ejemplos/sample.mp4')
ok, frame = ejemplo.read()
if not ok:
    print("No se pudo leer el video.")
    ejemplo.release() 
    cv2.destroyAllWindows()
    exit()

# el modelo necesita que seleccionemos la región de interés (ROI) para evitar falsas detecciones
ROI = cv2.selectROI('Selecciona la region de interes y presiona ENTER o SPACE', frame, fromCenter=False, showCrosshair=False)
cv2.destroyWindow('Selecciona la region de interes y presiona ENTER o SPACE')
x_roi, y_roi, w_roi, h_roi = ROI


while ejemplo.isOpened():
    ok, frame = ejemplo.read()
    if not ok:
        print("No se pudo leer el video o se terminó.")
        break

    frame_number += 1
    roi_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy() # recortar la región de interés

    """ PASO 2: DETECCIÓN Y TRACK DE VEHÍCULOS EN LA ROI """
    results = yolo_model.track(roi_frame, persist=True, tracker=tracker, classes=list(vehicles.keys())) # deteccion de vehiculos + track le añade un id unico a cada vehiculo
                                                        #usa el tracker + usa solo la lista de vehiculos que queremos detectar

    vehicle_tracks = {} # diccionario para almacenar las posiciones de los vehículos detectados en este frame
    if results[0].boxes.id is not None:
        for box, class_id, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
            class_id = int(class_id)
            track_id = int(track_id)
            x1, y1, x2, y2 = box.cpu().numpy() # box nos da las coordenadas del vehiculo
            x1 += x_roi
            y1 += y_roi
            x2 += x_roi
            y2 += y_roi
            vehicle_tracks[track_id] = (x1, y1, x2, y2)

    """ PASO 3: DETECCIÓN DE MATRÍCULAS EN LOS VEHÍCULOS DETECTADOS """
    license_plates = license_plates_detector(roi_frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1,y1,x2,y2,conf, class_id = license_plate
        x1 += x_roi
        y1 += y_roi
        x2 += x_roi
        y2 += y_roi

        for track_id, (carx1, cary1, carx2, cary2) in vehicle_tracks.items():
            # comprobar si la placa está dentro del vehículo
            if x1 > carx1 and y1 > cary1 and x2 < carx2 and y2 < cary2:
                plate_cut = frame[int(y1):int(y2), int(x1):int(x2)]
                plate_cut = cv2.resize(plate_cut, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                plate_gray = cv2.cvtColor(plate_cut, cv2.COLOR_BGR2GRAY) #en escala de grises para detectar mejor
                plate_rgb = cv2.cvtColor(plate_gray, cv2.COLOR_GRAY2RGB) # PaddleOCR necesita 3 canales
                ocr_result = ocr.predict(plate_rgb)

                plate_text = ""

                if ocr_result:
                    for res in ocr_result:
                        texts = res.get("rec_texts", [])
                        scores = res.get("rec_scores", [])

                        for text, score in zip(texts, scores):
                            if score > 0.7 and text:
                                plate_text = text.upper().replace(" ", "")
                                plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)
                                if len(plate_text) < 5:
                                  plate_text = ""


                if track_id not in vehicle_plates or len(plate_text) > len(vehicle_plates[track_id]):
                    vehicle_plates[track_id] = plate_text
                
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_number, track_id, vehicle_plates[track_id]])

                cv2.rectangle(frame, (int(carx1), int(cary1)), (int(carx2), int(cary2)), (255, 0, 0), 2) # dibujar rectángulo
                putTextRect(frame, f"ID:{track_id}", (int(carx1), int(cary1)-10), scale=1, thickness=1, colorR=(255,0,0), colorB=(255,255,255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # dibujar rectángulo
                putTextRect(frame, f"Plate: {vehicle_plates[track_id]}", (int(x1), int(y1) - 10), scale=1, thickness=1, colorR=(0,0,255), colorB=(255,255,255))
                print("PLATE FINAL:", plate_text)

    cv2.imshow('Deteccion de vehiculos y placas', frame) # mostrar el frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # salir con 'q' o escape
        break

# liberar espacio

ejemplo.release() 
cv2.destroyAllWindows()