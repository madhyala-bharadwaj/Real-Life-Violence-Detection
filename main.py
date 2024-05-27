from feature_extraction import *
from keras.models import load_model

model = load_model('model/best_model.keras')

cap = cv2.VideoCapture(0)

sequence_length = 16

while True:
    frames = []
    for _ in range(sequence_length):
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    if len(frames) < sequence_length:
        print("Not enough frames. Exiting...")
        break

    features = np.array([feature_extractor(frame) for frame in frames])
    features_reshaped = features.reshape((1, sequence_length, 2048))

    prediction = model.predict(features_reshaped)
    violence_probability = prediction[0][0]*100

    cv2.putText(frames[-1], f"Violence Prob: {violence_probability:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 0), 2)

    cv2.imshow("Live Video", frames[-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
