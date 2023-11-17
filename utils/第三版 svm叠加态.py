# 导入必要的库
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from elpv_reader import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.decomposition import PCA
import os

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
num_classes = 4  # 修改为四个类别
learning_rate = 0.0001

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
# Apply mapping (as before)
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# Split the dataset into mono and poly (or other types if applicable)
X_mono = images[types == 'mono']
y_mono = probs_mapped[types == 'mono']
X_poly = images[types == 'poly']
y_poly = probs_mapped[types == 'poly']

# Now, split each subset into training and testing sets
X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(
    X_mono, y_mono, test_size=0.25, stratify=y_mono, random_state=None)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_poly, test_size=0.25, stratify=y_poly, random_state=None)

# You can concatenate these subsets if you want a mixed training and test set
X_train = np.concatenate((X_train_mono, X_train_poly), axis=0)
y_train = np.concatenate((y_train_mono, y_train_poly), axis=0)
X_test = np.concatenate((X_test_mono, X_test_poly), axis=0)
y_test = np.concatenate((y_test_mono, y_test_poly), axis=0)

# Compute class weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Preprocessing function for images
def preprocess_image(image):
    img = Image.fromarray(image)
    if img.mode == 'L':
        img = img.convert('RGB')
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
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
y_test_mono = tf.keras.utils.to_categorical(y_test_mono, num_classes)
y_test_poly = tf.keras.utils.to_categorical(y_test_poly, num_classes)
# 定义 ResNet50 模型，只定义一次
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu', kernel_regularizer=l1_l2(l1=0.07, l2=0.08))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.03, l2=0.03))(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型，只编译一次
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# 添加 L1-L2 正则化
x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Check Point
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with the checkpoint callback
try:
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), sample_weight=sample_weights, callbacks=[checkpoint])
except Exception as e:
    print('Exception occurred: ', str(e))

# Load the best model
model.load_weights('best_model.h5')

# Define a function to evaluate the deep learning model
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
print("Evaluating Deep Learning Model on Mixed Data (Mono and Poly):")
evaluate_model(model, X_test, y_test)

print("\nEvaluating Deep Learning Model on Mono Data:")
evaluate_model(model, X_test_mono, y_test_mono)

print("\nEvaluating Deep Learning Model on Poly Data:")
evaluate_model(model, X_test_poly, y_test_poly)

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
# SVM Classifier with KFold Cross-Validation and training on the entire set
# Define feature extractor
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

# Feature extraction
train_features = feature_extractor.predict(X_train)
train_features = np.reshape(train_features, (train_features.shape[0], -1))

# 交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
svm_accuracies = []

for train_index, val_index in kf.split(train_features):
    # Split data for this fold
    X_train_fold, X_val_fold = train_features[train_index], train_features[val_index]
    y_train_fold, y_val_fold = np.argmax(y_train, axis=1)[train_index], np.argmax(y_train, axis=1)[val_index]

    # Train SVM Classifier
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm_classifier.fit(X_train_fold, y_train_fold)

    # Evaluate SVM Classifier
    svm_predictions = svm_classifier.predict(X_val_fold)
    svm_accuracy = accuracy_score(y_val_fold, svm_predictions)
    svm_accuracies.append(svm_accuracy)

# Train SVM Classifier on entire training set for final evaluation
svm_classifier.fit(train_features, np.argmax(y_train, axis=1))

# Extract features for each test set
test_features = feature_extractor.predict(X_test)
test_features = np.reshape(test_features, (test_features.shape[0], -1))
test_features_mono = feature_extractor.predict(X_test_mono)
test_features_mono = np.reshape(test_features_mono, (test_features_mono.shape[0], -1))
test_features_poly = feature_extractor.predict(X_test_poly)
test_features_poly = np.reshape(test_features_poly, (test_features_poly.shape[0], -1))

# Define a function to evaluate SVM
def evaluate_svm(svm_classifier, features, labels):
    svm_predictions = svm_classifier.predict(features)
    labels = np.argmax(labels, axis=1)  # Convert one-hot to label
    accuracy = accuracy_score(labels, svm_predictions)
    f1 = f1_score(labels, svm_predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, svm_predictions)

    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

# Evaluate SVM on different subsets
print("Evaluating SVM on Mixed Data (Mono and Poly):")
evaluate_svm(svm_classifier, test_features, y_test)

print("\nEvaluating SVM on Mono Data:")
evaluate_svm(svm_classifier, test_features_mono, y_test_mono)

print("\nEvaluating SVM on Poly Data:")
evaluate_svm(svm_classifier, test_features_poly, y_test_poly)
# Creating the meta-dataset
resnet_test_predictions = model.predict(X_test)
resnet_test_classes = np.argmax(resnet_test_predictions, axis=1)
svm_test_predictions = svm_classifier.predict(test_features)
meta_features = np.vstack([resnet_test_classes, svm_test_predictions]).T

# Train logistic regression meta model
meta_model = LogisticRegression()
meta_model.fit(meta_features, np.argmax(y_test, axis=1))

# Make final predictions on test set
final_predictions = meta_model.predict(meta_features)

# Evaluate final predictions
final_accuracy = accuracy_score(np.argmax(y_test, axis=1), final_predictions)
final_f1 = f1_score(np.argmax(y_test, axis=1), final_predictions, average='weighted')
print(f'Final Meta-Model Accuracy: {final_accuracy}')
print(f'Final Meta-Model F1 Score: {final_f1}')
