import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(r"C:\Users\user\Desktop\eye_opacity_detector\models\corneal_opacity_classifier.h5")
image_path = "C:\\Users\\user\\Desktop\\NPK\\eye_opacity_detector\\data\\test_eye1.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Не удалось загрузить изображение")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (128, 128))
img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
prob = prediction[0][0]

label = (
    "Обнаружены признаки помутнения"
    if prob > 0.5
    else "Признаки помутнения не обнаружены"
)

print(f"Результат: {label}")
print(f"Вероятность помутнения: {prob*100:.2f}%")

plt.imshow(img)
plt.title(f"{label} ({prob*100:.1f}%)")
plt.axis("off")
plt.show()
