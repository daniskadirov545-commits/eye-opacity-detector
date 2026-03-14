import os
import cv2
import matplotlib.pyplot as plt

normal_path = r"C:\Users\user\Desktop\eye_opacity_detector\data\normal_eyes"
opacity_path = r"C:\Users\user\Desktop\eye_opacity_detector\data\other_opacities"

normal_files = os.listdir(normal_path)
opacity_files = os.listdir(opacity_path)

print(f"Норма: {len(normal_files)}")
print(f"Помутнения: {len(opacity_files)}")

img_normal = cv2.imread(os.path.join(normal_path, normal_files[0]))
img_opacity = cv2.imread(os.path.join(opacity_path, opacity_files[0]))

img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
img_opacity = cv2.cvtColor(img_opacity, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img_normal)
plt.title("Норма")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_opacity)
plt.title("Помутнение роговицы")
plt.axis("off")

plt.show()
