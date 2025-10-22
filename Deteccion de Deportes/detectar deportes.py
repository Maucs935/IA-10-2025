

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt



dirname = r"C:\Users\mauri\Downloads\Datadep\sportimages"
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ""

print("Ruta actual del dataset:", imgpath)

# Recorre el dataset y cuenta imágenes
for root, dirs, files in os.walk(dirname):
    for name in files:
        if name.lower().endswith((".jpg", ".png", ".jpeg")):
            images.append(os.path.join(root, name))
    for name in dirs:
        directories.append(os.path.join(root, name))

print(f"Total de carpetas: {len(directories)}")
print(f"Total de imágenes: {len(images)}")



batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
    dirname,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dirname,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)



AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



num_classes = len(class_names)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.legend(loc='lower right')
plt.title('Precisión del modelo')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.legend(loc='upper right')
plt.title('Pérdida del modelo')
plt.show()


#Guardar el modelo entrenado


model.save("sports_model.h5")

print("✅ Entrenamiento completado y modelo guardado como 'sports_model.h5'")



# 🧪 8. (Opcional) Probar con una imagen individual


from tensorflow.keras.preprocessing import image
import numpy as np

# Cambia esta ruta a una imagen de prueba
img_path = r"C:\Users\mauri\Downloads\Datadep\sportimages\football\ejemplo.jpg"

if os.path.exists(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("\n🔍 Predicción:")
    print(f"Clase predicha: {class_names[np.argmax(score)]}")
    print(f"Confianza: {100 * np.max(score):.2f}%")
else:
    print("\n⚠️ No se encontró la imagen de prueba. Verifica la ruta del archivo.")
