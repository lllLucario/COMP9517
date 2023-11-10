import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib import pyplot as plt
from elpv_reader import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
import os

# Parameters
data_csv_path = 'labels.csv'  # Path to CSV file
image_directory = './images'  # Image folder path
batch_size = 32  # Batch size
target_size = (224, 224)  # Image target size
num_epochs = 10  # Number of training epochs
num_classes = 4  # Number of classes
learning_rate = 0.0001  # Learning rate

# Load dataset
images, proba, types = load_dataset()

# Map probabilities to class labels
def map_probability_to_class(prob):
    if prob == 0:
        return 0  # Fully functional
    elif prob <= 0.33:
        return 1  # Possibly defective
    elif prob <= 0.67:
        return 2  # Likely defective
    else:
        return 3  # Certainly defective

# Apply mapping
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, probs_mapped, test_size=0.25, stratify=probs_mapped)

# Compute class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Preprocessing function for images
def preprocess_image(image):
    img = Image.fromarray(image)

    # Check if image is grayscale; if so, convert to RGB
    if img.mode == 'L':
        img = img.convert('RGB')

    # Enhance contrast, apply median filter, and resize
    img = ImageEnhance.Contrast(img).enhance(2)  # Adjust contrast
    img = img.filter(ImageFilter.MedianFilter(size=3))  # Apply median filter
    img = img.resize(target_size)  # Resize image
    img_array = np.array(img) / 255.0  # Normalize image

    return img_array

# Apply preprocessing to the images
X_train = np.array([preprocess_image(img) for img in X_train])
X_test = np.array([preprocess_image(img) for img in X_test])

# Ensure labels are in one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Check Point
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model with the checkpoint callback
try:
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test),
                        sample_weight=sample_weights, callbacks=[checkpoint])
except Exception as e:
    print('Exception occurred: ', str(e))

# Save the final model
model.save('final_model.h5')

# Evaluate model
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
accuracy = accuracy_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Random Forest classifier
# Feature extraction for Random Forest
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

# Extract features
train_features = feature_extractor.predict(X_train)
test_features = feature_extractor.predict(X_test)

# Reshape features for Random Forest compatibility
train_features = np.reshape(train_features, (train_features.shape[0], -1))
test_features = np.reshape(test_features, (test_features.shape[0], -1))

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, np.argmax(y_train, axis=1))

# Evaluate Random Forest classifier
rf_predictions = rf_classifier.predict(test_features)
rf_accuracy = accuracy_score(np.argmax(y_test, axis=1), rf_predictions)
rf_f1 = f1_score(np.argmax(y_test, axis=1), rf_predictions, average='weighted')

print(f'Random Forest Confusion Matrix:\n{confusion_matrix(np.argmax(y_test, axis=1), rf_predictions)}')
print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'Random Forest F1 Score: {rf_f1}')