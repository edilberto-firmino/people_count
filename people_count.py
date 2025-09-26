from ultralytics import YOLO
import cv2

# Carregar o modelo YOLO pré-treinado (pessoas = classe 0)
model = YOLO("yolov8n.pt")  # n = versão mais leve

# Abrir a webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fazer a inferência
    results = model(frame, verbose=False)

    # Contar apenas objetos da classe "pessoa" (ID 0 no COCO dataset)
    people_count = 0
    for box in results[0].boxes:
        if int(box.cls) == 0:
            people_count += 1

    # Desenhar caixas e texto
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Pessoas: {people_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar a imagem
    cv2.imshow("Contagem de Pessoas", annotated_frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
