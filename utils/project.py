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
num_epochs = 10
num_classes = 4  # 修改为四个类别
learning_rate = 0.0001

# Load dataset
images, proba, types = load_dataset()

# Map probabilities to class labels
def map_probability_to_class(prob):
    if prob < 0.33:
        return 0
    elif prob < 0.67:
        return 1
    elif prob < 1.0:
        return 2
    else:
        return 3

# Apply mapping
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# 分割数据集为单晶、多晶和混合
X_mono, y_mono = images[types == 'mono'], probs_mapped[types == 'mono']
X_poly, y_poly = images[types == 'poly'], probs_mapped[types == 'poly']
X_mixed, y_mixed = images, probs_mapped

# Split data for mixed types
X_train, X_test, y_train, y_test = train_test_split(X_mixed, y_mixed, test_size=0.25, stratify=y_mixed)

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

# 确保标签是独热编码格式
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 定义 ResNet50 模型，只定义一次
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
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

# SVM 分类器
# 交叉验证参数
resnet_test_predictions = model.predict(X_test)
resnet_test_classes = np.argmax(resnet_test_predictions, axis=1)

# 初始化用于保存交叉验证预测的数组
svm_cv_predictions = np.zeros(len(X_test))

# 交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
svm_accuracies = []
svm_cv_predictions = np.zeros(len(X_test))

for train_index, test_index in kf.split(X_train):
    # 分割数据
    X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

    # 特征提取
    train_features = feature_extractor.predict(X_train_fold)
    train_features = np.reshape(train_features, (train_features.shape[0], -1))
    val_features = feature_extractor.predict(X_val_fold)
    val_features = np.reshape(val_features, (val_features.shape[0], -1))

    # 训练 SVM 分类器
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm_classifier.fit(train_features, np.argmax(y_train_fold, axis=1))

    # 在验证集上预测
    svm_val_predictions = svm_classifier.predict(val_features)

    # 获取一维数组格式的标签
    y_val_fold_labels = np.argmax(y_val_fold, axis=1)

    # 计算并存储 SVM 准确率
    svm_accuracy = accuracy_score(y_val_fold_labels, svm_val_predictions)

    svm_accuracies.append(svm_accuracy)

# 创建元数据集
meta_features = np.vstack([resnet_test_classes, svm_cv_predictions]).T

# 训练逻辑回归元模型
meta_model = LogisticRegression()
meta_model.fit(meta_features, np.argmax(y_test, axis=1))

# 对测试集进行最终预测
final_predictions = meta_model.predict(meta_features)

# 性能评估
final_accuracy = accuracy_score(np.argmax(y_test, axis=1), final_predictions)
final_f1 = f1_score(np.argmax(y_test, axis=1), final_predictions, average='weighted')

# 输出平均性能指标
print(f'Final Accuracy: {final_accuracy}')
print(f'Final F1 Score: {final_f1}')