import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint

# 自定义损失函数
def custom_loss(y_true, y_pred):
    absolute_diff = tf.abs(y_true - y_pred)
    nonzero_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    loss = tf.reduce_sum(tf.multiply(absolute_diff, nonzero_mask))
    loss += 0.1*tf.reduce_sum(tf.multiply(absolute_diff, zero_mask))
    return loss

def precision_score_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 计算真阳性（True Positive）
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 计算假阳性（False Positive）

    precision = tp / (tp + fp + 1e-10)  # 计算精确度，添加平滑项防止分母为 0

    return precision

def recall_score_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 计算真阳性（True Positive）
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 计算假阴性（False Negative）
    recall = tp / (tp + fn + 1e-10)  # 计算召回率，添加平滑项防止分母为 0
    return recall

def f1_score_binary(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 计算真阳性（True Positive）
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 计算假阳性（False Positive）
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 计算假阴性（False Negative）

    precision = tp / (tp + fp + 1e-10)  # 计算精确度，添加平滑项防止分母为 0
    recall = tp / (tp + fn + 1e-10)  # 计算召回率，添加平滑项防止分母为 0

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  # 计算 F1 分数，添加平滑项防止分母为 0

    return f1

X = np.load('../embedding/X_decay.npy')
y = np.load('../embedding/y_scores.npy')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(y_train.shape[1])
])


adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=adam_optimizer, loss=custom_loss)


# 定义 ModelCheckpoint 回调函数
checkpoint_filepath = 'LSTM_decay_model.keras'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      save_weights_only=False,
                                      monitor='val_loss',
                                      mode='min',
                                      save_best_only=True)


history = model.fit(X_train, y_train, epochs=30, batch_size=5, validation_split=0.1, verbose=1, callbacks=[checkpoint_callback])


train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
