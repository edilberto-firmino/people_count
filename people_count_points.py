from ultralytics import YOLO
import cv2

# Carregar modelo leve do YOLOv8
model = YOLO("yolov8n.pt")

# Abrir webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar detecção
    results = model(frame, verbose=False)

    # Contagem e desenho dos pontos
    people_count = 0
    for box in results[0].boxes:
        if int(box.cls) == 0:  # classe "pessoa"
            people_count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # ponto magenta

    # Mostrar total
    cv2.putText(frame, f"Pessoas: {people_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Exibir janela
    cv2.imshow("Contagem de Pessoas - Pontos", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()