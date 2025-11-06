import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os

# Load training and testing datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    batch_size=32
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32
)

# Normalize the data
class_names = train_dataset.class_names
normalization_layer = layers.Rescaling(1. / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Visualize sample training images
plt.figure(figsize=(8, 8))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# CNN model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.summary()

# Compile and train 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

# Accuracy and Loss graphs 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize learned filters from the first layer
first_layer = model.layers[0]
weights, biases = first_layer.get_weights()
print("Shape of filter weights:", weights.shape)

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[:, :, 0, i], cmap='gray')
    ax.set_title(f'Filter {i+1}')
    ax.axis('off')
plt.show()

# Test the images 
samples_folder = "dataset/samples"
for img_name in os.listdir(samples_folder):
    img_path = os.path.join(samples_folder, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    print(f"\nImage: {img_name}")
    print("Predicted Class:", class_names[predicted_class_index])
    print("Confidence: {:.2f}%".format(confidence))
    plt.imshow(img)
    plt.title(f"{img_name}\nPrediction: {class_names[predicted_class_index]} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
