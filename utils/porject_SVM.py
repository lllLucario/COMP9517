import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from matplotlib import pyplot as plt
from elpv_reader import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
import os

# Parameters
data_csv_path = 'labels.csv'
image_directory = './images'
batch_size = 32
target_size = (224, 224)
num_epochs = 75
num_classes = 4
learning_rate = 0.00005

# Load dataset
images, proba, types = load_dataset()

# Map probabilities to class labels
def map_probability_to_class(prob):
    if prob == 0:
        return 0
    elif prob <= 0.33:
        return 1
    elif prob <= 0.67:
        return 2
    else:
        return 3

# Apply mapping
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, probs_mapped, test_size=0.25, stratify=probs_mapped, random_state=None)

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

# Ensure labels are in one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Perform data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the ImageDataGenerator to the training data
datagen.fit(X_train)

# Augment the training data by using the ImageDataGenerator
augmented_data = datagen.flow(X_train, y_train, batch_size=batch_size)

# Combine the original and augmented data
X_train_augmented = np.concatenate([X_train, augmented_data[0][0]])
y_train_augmented = np.concatenate([y_train, augmented_data[0][1]])

# Recompute class weights for the combined dataset
sample_weights_augmented = compute_sample_weight(class_weight='balanced', y=np.argmax(y_train_augmented, axis=1))


# Define ResNet50 model with Elastic Net regularization (L1 and L2)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)  # Increase dropout rate to 0.7
# Add L1-L2 regularization to the Dense layer
x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)  # Increase dropout rate to 0.7
# Add L1-L2 regularization to the output layer as well
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Check Point
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with the checkpoint callback
try:
    history = model.fit(
        X_train_augmented, y_train_augmented,
        sample_weight=sample_weights_augmented,
        epochs=num_epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
except Exception as e:
    print('Exception occurred: ', str(e))

# Load the best model
model.load_weights('best_model.h5')

# Evaluate model on test set
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

rf_classifier = RandomForestClassifier(
    n_estimators=200,            # 树的数量
    max_depth=30,                # 树的最大深度
    min_samples_split=10,        # 内部节点再划分所需最小样本数
    min_samples_leaf=8,          # 叶子节点最少样本数
    max_features='sqrt',         # 寻找最佳分割时考虑的最大特征数
    bootstrap=False,             # 是否使用bootstrap样本
    class_weight='balanced',     # 类别权重
    random_state=80            # 随机种子
)

# 训练随机森林分类器
rf_classifier.fit(train_features, np.argmax(y_train, axis=1))

# 评估随机森林分类器
rf_predictions = rf_classifier.predict(test_features)
rf_accuracy = accuracy_score(np.argmax(y_test, axis=1), rf_predictions)
rf_f1 = f1_score(np.argmax(y_test, axis=1), rf_predictions, average='weighted')
print(f'Random Forest Confusion Matrix:\n{confusion_matrix(np.argmax(y_test, axis=1), rf_predictions)}')
print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'Random Forest F1 Score: {rf_f1}')