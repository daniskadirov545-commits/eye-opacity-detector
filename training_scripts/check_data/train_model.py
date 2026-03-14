import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from prepare_data import X_train, X_test, y_train, y_test

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nТочность модели: {acc*100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nОтчёт о классификации:\n")
print(classification_report(
    y_test, y_pred,
    target_names=["Норма", "Помутнение"]
))

model.save(
    r"C:\Users\user\Desktop\eye_opacity_detector\models\corneal_opacity_classifier.h5"
)

print("\n✅ Модель сохранена")
