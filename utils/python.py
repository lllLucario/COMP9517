# 导入必要的库
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from elpv_reader import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
num_epochs = 20
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
# Apply mapping
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# 分割数据集为单晶、多晶和所有类型
X_mono, y_mono = images[types == 'mono'], probs_mapped[types == 'mono']
X_poly, y_poly = images[types == 'poly'], probs_mapped[types == 'poly']
X_all, y_all = images, probs_mapped

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


# Split and preprocess the datasets
X_train_mono, X_test_mono, y_train_mono, y_test_mono = train_test_split(X_mono, y_mono, test_size=0.2, random_state=42)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Apply preprocessing to the images
X_train_mono = np.array([preprocess_image(img) for img in X_train_mono])
X_test_mono = np.array([preprocess_image(img) for img in X_test_mono])
X_train_poly = np.array([preprocess_image(img) for img in X_train_poly])
X_test_poly = np.array([preprocess_image(img) for img in X_test_poly])
X_train_all = np.array([preprocess_image(img) for img in X_train_all])
X_test_all = np.array([preprocess_image(img) for img in X_test_all])

# Ensure labels are in one-hot encoding format
y_train_mono = tf.keras.utils.to_categorical(y_train_mono, num_classes)
y_test_mono = tf.keras.utils.to_categorical(y_test_mono, num_classes)
y_train_poly = tf.keras.utils.to_categorical(y_train_poly, num_classes)
y_test_poly = tf.keras.utils.to_categorical(y_test_poly, num_classes)
y_train_all = tf.keras.utils.to_categorical(y_train_all, num_classes)
y_test_all = tf.keras.utils.to_categorical(y_test_all, num_classes)


# 定义 ResNet50 模型
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 创建模型
model_mono = create_model()
model_poly = create_model()
model_all = create_model()

# 编译模型
model_mono.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model_poly.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model_all.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Check Point
checkpoint_mono = ModelCheckpoint('best_model_mono.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_poly = ModelCheckpoint('best_model_poly.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_all = ModelCheckpoint('best_model_all.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train model
history_mono = model_mono.fit(X_train_mono, y_train_mono, epochs=num_epochs, validation_data=(X_test_mono, y_test_mono), sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_mono), callbacks=[checkpoint_mono])
history_poly = model_poly.fit(X_train_poly, y_train_poly, epochs=num_epochs, validation_data=(X_test_poly, y_test_poly), sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_poly), callbacks=[checkpoint_poly])
history_all = model_all.fit(X_train_all, y_train_all, epochs=num_epochs, validation_data=(X_test_all, y_test_all), sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_all), callbacks=[checkpoint_all])

# Load the best model
model_mono.load_weights('best_model_mono.h5')
model_poly.load_weights('best_model_poly.h5')
model_all.load_weights('best_model_all.h5')

# 评估模型并创建混淆矩阵
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    return conf_matrix, accuracy, f1

# 评估单晶模型
conf_matrix_mono, accuracy_mono, f1_mono = evaluate_model(model_mono, X_test_mono, y_test_mono)
print(f'Monocrystalline - Confusion Matrix:\n{conf_matrix_mono}\nAccuracy: {accuracy_mono}\nF1 Score: {f1_mono}')

# 评估多晶模型
conf_matrix_poly, accuracy_poly, f1_poly = evaluate_model(model_poly, X_test_poly, y_test_poly)
print(f'Polycrystalline - Confusion Matrix:\n{conf_matrix_poly}\nAccuracy: {accuracy_poly}\nF1 Score: {f1_poly}')

# 评估整体模型
conf_matrix_all, accuracy_all, f1_all = evaluate_model(model_all, X_test_all, y_test_all)
print(f'Overall - Confusion Matrix:\n{conf_matrix_all}\nAccuracy: {accuracy_all}\nF1 Score: {f1_all}')

# 定义特征提取器为 ResNet50 的最后一个卷积层的输出
feature_extractor = Model(inputs=model_all.input, outputs=model_all.get_layer('conv5_block6_out').output)

def perform_svm_classification(X_train, y_train, X_test, y_test):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    svm_accuracies = []
    svm_cv_predictions_full = np.zeros(len(X_train))

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        train_features = feature_extractor.predict(X_train_fold)
        train_features = np.reshape(train_features, (train_features.shape[0], -1))
        val_features = feature_extractor.predict(X_val_fold)
        val_features = np.reshape(val_features, (val_features.shape[0], -1))

        pca = PCA(n_components=200)
        train_features_pca = pca.fit_transform(train_features)
        val_features_pca = pca.transform(val_features)

        svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')
        svm_classifier.fit(train_features_pca, np.argmax(y_train_fold, axis=1))

        svm_val_predictions = svm_classifier.predict(val_features_pca)
        svm_cv_predictions_full[val_index] = svm_val_predictions

        y_val_fold_labels = np.argmax(y_val_fold, axis=1)
        svm_accuracy = accuracy_score(y_val_fold_labels, svm_val_predictions)
        svm_accuracies.append(svm_accuracy)

    svm_avg_accuracy = np.mean(svm_accuracies)
    return svm_cv_predictions_full, svm_avg_accuracy

# Perform SVM classification for monocrystalline, polycrystalline, and all types
svm_predictions_mono, svm_accuracy_mono = perform_svm_classification(X_train_mono, y_train_mono, X_test_mono, y_test_mono)
svm_predictions_poly, svm_accuracy_poly = perform_svm_classification(X_train_poly, y_train_poly, X_test_poly, y_test_poly)
svm_predictions_all, svm_accuracy_all = perform_svm_classification(X_train_all, y_train_all, X_test_all, y_test_all)

print(f'SVM - Monocrystalline Average Accuracy: {svm_accuracy_mono}')
print(f'SVM - Polycrystalline Average Accuracy: {svm_accuracy_poly}')
print(f'SVM - Overall Average Accuracy: {svm_accuracy_all}')

# 创建和训练元模型
def create_and_train_meta_model(X_train, y_train, X_test, y_test, svm_predictions):
    test_indices = range(len(X_train) - len(X_test), len(X_train))
    meta_features = np.vstack([np.argmax(model_all.predict(X_test), axis=1), svm_predictions[test_indices]]).T

    meta_model = LogisticRegression()
    meta_model.fit(meta_features, np.argmax(y_test, axis=1))

    final_predictions = meta_model.predict(meta_features)
    final_accuracy = accuracy_score(np.argmax(y_test, axis=1), final_predictions)
    final_f1 = f1_score(np.argmax(y_test, axis=1), final_predictions, average='weighted')
    return final_accuracy, final_f1

# 训练并评估元模型
final_accuracy_mono, final_f1_mono = create_and_train_meta_model(X_train_mono, y_train_mono, X_test_mono, y_test_mono, svm_predictions_mono)
final_accuracy_poly, final_f1_poly = create_and_train_meta_model(X_train_poly, y_train_poly, X_test_poly, y_test_poly, svm_predictions_poly)
final_accuracy_all, final_f1_all = create_and_train_meta_model(X_train_all, y_train_all, X_test_all, y_test_all, svm_predictions_all)

print(f'Meta Model - Monocrystalline Final Accuracy: {final_accuracy_mono}, Final F1 Score: {final_f1_mono}')
print(f'Meta Model - Polycrystalline Final Accuracy: {final_accuracy_poly}, Final F1 Score: {final_f1_poly}')
print(f'Meta Model - Overall Final Accuracy: {final_accuracy_all}, Final F1 Score: {final_f1_all}')
