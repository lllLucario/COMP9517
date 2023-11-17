import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, BatchNormalization,
                                     Dropout, Multiply, Reshape)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from elpv_reader import load_dataset  # Assuming this is a custom module
import  random
# SE Block definition
def se_block(input_feature, ratio=16):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    se_feature = Dense(channel // ratio, activation='relu')(se_feature)
    se_feature = Dense(channel, activation='sigmoid')(se_feature)
    return Multiply()([input_feature, se_feature])

# Parameters
data_csv_path = 'labels.csv'
image_directory = './images'
batch_size = 64
target_size = (224, 224)
num_epochs = 80
num_classes = 4
learning_rate = 0.0001
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)


# Load dataset
images, proba, types = load_dataset()

# Map probabilities to class labels
def map_probability_to_class(prob):
    if prob == 0:
        return 0
    elif prob <= 0.34:
        return 1
    elif prob <= 0.68:
        return 2
    else:
        return 3

# Apply mapping
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, probs_mapped, test_size=0.25, stratify=probs_mapped)

# Split test set further into mono and poly
X_test_mono = [X_test[i] for i in range(len(X_test)) if types[i] == 'mono']
y_test_mono = [y_test[i] for i in range(len(y_test)) if types[i] == 'mono']
X_test_poly = [X_test[i] for i in range(len(X_test)) if types[i] == 'poly']
y_test_poly = [y_test[i] for i in range(len(y_test)) if types[i] == 'poly']

# Compute class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Preprocessing function for images
def preprocess_image(image):
    img = Image.fromarray(image)
    if img.mode == 'L':
        img = img.convert('RGB')

    # Adjust contrast
    contrast_factor = random.uniform(0.5, 1.5)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Adjust brightness
    brightness_factor = random.uniform(0.65, 1.35)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    img = ImageEnhance.Contrast(img).enhance(2)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

# Apply preprocessing to the images
X_train = np.array([preprocess_image(img) for img in X_train])
X_test = np.array([preprocess_image(img) for img in X_test])
X_test_mono = np.array([preprocess_image(img) for img in X_test_mono])
X_test_poly = np.array([preprocess_image(img) for img in X_test_poly])

# Ensure labels are one-hot encoded
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test = tf.keras.utils.to_categorical(y_test, 4)
y_test_mono = tf.keras.utils.to_categorical(y_test_mono, 4)
y_test_poly = tf.keras.utils.to_categorical(y_test_poly, 4)

# Define ResNet50 model with Elastic Net regularization (L1 and L2)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)  # Increase dropout rate to 0.7
x = Dense(2048, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)  # Increase dropout rate to 0.7
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Check Point
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with the checkpoint callback
history = None
try:
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test),
                        sample_weight=sample_weights, callbacks=[checkpoint, reduce_lr])
except Exception as e:
    print('Exception occurred during training: ', str(e))

# Load the best model
model.load_weights('best_model.h5')

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

# Evaluate on different subsets
print("Evaluating on Mixed Data (Mono and Poly):")
evaluate_model(model, X_test, y_test)

print("\nEvaluating on Mono Data:")
evaluate_model(model, X_test_mono, y_test_mono)

print("\nEvaluating on Poly Data:")
evaluate_model(model, X_test_poly, y_test_poly)

# Plot training & validation accuracy and loss values
if history:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# SVM Classifier with KFold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

svm_accuracies = []
svm_f1_scores = []

# ... [前面的代码，包括模型训练等] ...

# Feature extraction for SVM
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
train_features = feature_extractor.predict(X_train)
train_features = np.reshape(train_features, (train_features.shape[0], -1))

# SVM Classifier with KFold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

svm_accuracies = []
svm_f1_scores = []

# KFold Cross-Validation
for train_index, val_index in kf.split(train_features):
    X_train_fold, X_val_fold = train_features[train_index], train_features[val_index]
    y_train_fold, y_val_fold = np.argmax(y_train, axis=1)[train_index], np.argmax(y_train, axis=1)[val_index]

    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm_classifier.fit(X_train_fold, y_train_fold)

    svm_predictions = svm_classifier.predict(X_val_fold)
    svm_accuracy = accuracy_score(y_val_fold, svm_predictions)
    svm_f1 = f1_score(y_val_fold, svm_predictions, average='weighted')

    svm_accuracies.append(svm_accuracy)
    svm_f1_scores.append(svm_f1)

print(f'SVM Average Accuracy: {np.mean(svm_accuracies)}')
print(f'SVM Average F1 Score: {np.mean(svm_f1_scores)}')

# Train SVM Classifier on entire training set for final evaluation
svm_classifier.fit(train_features, np.argmax(y_train, axis=1))

# Extract features for each test set
test_features = feature_extractor.predict(X_test)
test_features = np.reshape(test_features, (test_features.shape[0], -1))

test_features_mono = feature_extractor.predict(X_test_mono)
test_features_mono = np.reshape(test_features_mono, (test_features_mono.shape[0], -1))

test_features_poly = feature_extractor.predict(X_test_poly)
test_features_poly = np.reshape(test_features_poly, (test_features_poly.shape[0], -1))

# Evaluate SVM Classifier
def evaluate_svm(svm_classifier, features, labels):
    svm_predictions = svm_classifier.predict(features)
    accuracy = accuracy_score(labels, svm_predictions)
    f1 = f1_score(labels, svm_predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, svm_predictions)

    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

print("Evaluating SVM on Mixed Data (Mono and Poly):")
evaluate_svm(svm_classifier, test_features, np.argmax(y_test, axis=1))

print("\nEvaluating SVM on Mono Data:")
evaluate_svm(svm_classifier, test_features_mono, np.argmax(y_test_mono, axis=1))

print("\nEvaluating SVM on Poly Data:")
evaluate_svm(svm_classifier, test_features_poly, np.argmax(y_test_poly, axis=1))
