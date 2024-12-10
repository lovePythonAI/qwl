import tensorflow as tf
from tensorflow.keras import layers, models

# 加载cifar100数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()


# 定义 Inception 模块
def inception_module(x):
    branch1x1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    branch3x3 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    branch3x3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    branch5x5 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    branch_pool = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(branch_pool)

    outputs = layers.Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])
    return outputs


# 构建 GoogleNet 模型
inputs = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (7, 7), padding='same', strides=2, activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

# 添加第一个Inception模块
x = inception_module(x)

# 添加第二个Inception模块
x = inception_module(x)

# 全局平均池化层
x = layers.GlobalAveragePooling2D()(x)

# 输出层
outputs = layers.Dense(100, activation='softmax')(x)

# 创建模型
model = models.Model(inputs, outputs)

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# 评估模型
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)
