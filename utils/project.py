import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from elpv_reader import load_dataset

# 参数设置
data_csv_path = 'labels.csv'  # CSV文件路径
image_directory = './images'  # 图片文件夹路径
batch_size = 32  # 批量大小
target_size = (224, 224)  # 图片目标大小
num_epochs = 10  # 训练周期数
num_classes = 4  # 类别数量
learning_rate = 0.0001  # 学习率

# 加载数据集
images, proba, types = load_dataset()

# 将概率映射到类别
def map_probability_to_class(prob):
    if prob == 0:
        return 0  # 完全正常
    elif prob <= 0.33:
        return 1  # 可能有缺陷
    elif prob <= 0.67:
        return 2  # 很可能有缺陷
    else:
        return 3  # 肯定有缺陷

# 映射概率到类别标签
probs_mapped = np.array([map_probability_to_class(prob) for prob in proba])

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(images, probs_mapped, test_size=0.25, stratify=probs_mapped)

# 类别权重计算
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)


# 首先，确认所有图像都是numpy数组
X_train = [np.array(img) for img in X_train]
X_test = [np.array(img) for img in X_test]

# 接下来，确保所有图像都是三通道的
X_train = [img if img.ndim == 3 else np.stack((img,)*3, axis=-1) for img in X_train]
X_test = [img if img.ndim == 3 else np.stack((img,)*3, axis=-1) for img in X_test]

# 使用tf.image.resize调整图像大小
X_train = np.stack([tf.image.resize(img, target_size).numpy() for img in X_train]) / 255.0
X_test = np.stack([tf.image.resize(img, target_size).numpy() for img in X_test]) / 255.0

# 确保标签是正确的one-hot编码格式
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 创建ResNet50模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 最终模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), sample_weight=sample_weights)

# 评估模型
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
accuracy = accuracy_score(true_classes, predicted_classes)
f1 = f1_score(true_classes, predicted_classes, average='weighted')

print(f'混淆矩阵:\n{conf_matrix}')
print(f'准确率: {accuracy}')
print(f'F1分数: {f1}')

# 随机森林分类器
# 提取特征用于随机森林
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)

# 生成特征
train_features = feature_extractor.predict(X_train)
test_features = feature_extractor.predict(X_test)

# 重塑特征以适应随机森林
train_features = np.reshape(train_features, (train_features.shape[0], -1))
test_features = np.reshape(test_features, (test_features.shape[0], -1))

# 训练随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features, np.argmax(y_train, axis=1))

# 评估随机森林分类器
rf_predictions = rf_classifier.predict(test_features)
rf_accuracy = accuracy_score(np.argmax(y_test, axis=1), rf_predictions)
rf_f1 = f1_score(np.argmax(y_test, axis=1), rf_predictions, average='weighted')

print(f'随机森林混淆矩阵:\n{confusion_matrix(np.argmax(y_test, axis=1), rf_predictions)}')
print(f'随机森林准确率: {rf_accuracy}')
print(f'随机森林F1分数: {rf_f1}')
