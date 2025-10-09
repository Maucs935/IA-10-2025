import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Carga el modelo entrenado
model = load_model("models/emotion_model.h5")
labels = ["angry", "disgust", "fear","happy","neutral", "sad","surprise"]

# Carga el detector de rostros
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi = cv.resize(roi_color, (224, 224))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        emotion = labels[np.argmax(preds)]
        confidence = preds[np.argmax(preds)] * 100

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv.putText(frame, f"{emotion}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv.imshow("Tas tiste o feli", frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv.destroyAllWindows()
