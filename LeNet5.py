# 导入 TensorFlow 库
import tensorflow as tf

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 构建卷积神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10)

# 评估模型
test_acc = model.evaluate(x_test, y_test, verbose=1)
