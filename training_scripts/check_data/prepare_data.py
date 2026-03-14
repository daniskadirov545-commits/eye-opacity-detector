import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# пути к данным
normal_path = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\normal_eyes"
opacity_path = r"C:\Users\user\Desktop\NPK\eye_opacity_detector\data\cataract_eyes"


IMG_SIZE = 128

def load_images_from_folder(folder, label):
    images = []
    labels = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels

# загрузка данных
normal_imgs, normal_labels = load_images_from_folder(normal_path, 0)
opacity_imgs, opacity_labels = load_images_from_folder(opacity_path, 1)

X = np.array(normal_imgs + opacity_imgs)
y = np.array(normal_labels + opacity_labels)

print("Всего изображений:", len(X))
print("Размер X:", X.shape)
print("Баланс классов:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
