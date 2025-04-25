import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Set dataset directory
data_dir = "C:\\Users\\DELL\\Desktop\\Sign GPT\\ISL Dataset"  # Change this path accordingly
img_size = 64

# Load dataset
def load_data():
    X, Y = [], []
    classes = sorted(os.listdir(data_dir))
    for label, sign in enumerate(classes):
        sign_path = os.path.join(data_dir, sign)
        for img_name in os.listdir(sign_path):
            img = cv2.imread(os.path.join(sign_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            Y.append(label)
    
    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    Y = np.array(Y)
    return X, Y, classes

# Load data
X, Y, class_names = load_data()
print(f"Loaded {len(X)} images across {len(class_names)} classes.")

# Save class labels
with open("class_labels.txt", "w") as f:
    f.write("\n".join(class_names))

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train = to_categorical(Y_train, num_classes=len(class_names))
Y_test = to_categorical(Y_test, num_classes=len(class_names))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=32),
    validation_data=(X_test, Y_test),
    epochs=50,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Save model and labels
model.save("isl_model.h5")

# Load model and class names for inference
model = keras.models.load_model("isl_model.h5")
with open("class_labels.txt", "r") as f:
    class_names = f.read().splitlines()

# Preprocessing for webcam
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
    return reshaped

# Live detection with webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)

    if confidence < 0.75:
        label = "Unclear or Non-Gesture Detected"
        color = (0, 0, 255)  # Red
    else:
        label = class_names[predicted_index]
        color = (0, 255, 0)  # Green

    # Display label
    cv2.putText(frame, f"{label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
