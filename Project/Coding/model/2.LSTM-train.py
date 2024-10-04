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

X = np.load('../embedding/X_scores.npy')
y = np.load('../embedding/y_scores.npy')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(y_train.shape[1])
])


adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=adam_optimizer, loss=custom_loss)


# 定义 ModelCheckpoint 回调函数
checkpoint_filepath = 'LSTM_model.keras'
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



