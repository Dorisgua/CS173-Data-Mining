import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

X = np.load('../embedding/X_decay.npy')
y = np.load('../embedding/y_scores.npy')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 86)),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(86)
])

adam_optimizer = optimizers.Adam(learning_rate=0.0005)
def custom_loss(y_true, y_pred):
    absolute_diff = tf.abs(y_true - y_pred)
    nonzero_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    loss = tf.reduce_sum(tf.multiply(absolute_diff, nonzero_mask))
    loss += 0.1 * tf.reduce_sum(tf.multiply(absolute_diff, zero_mask))
    return loss
checkpoint_filepath = 'CNN_LSTM_decay_model.keras'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                      save_weights_only=False,
                                      monitor='val_loss',
                                      mode='min',
                                      save_best_only=True)# 编译模型，使用自定义损失函数
model.compile(optimizer=adam_optimizer, loss=custom_loss)

model_checkpoint = callbacks.ModelCheckpoint(filepath='model_epoch_{epoch:02d}.keras', save_best_only=False)

history = model.fit(X_train, y_train, epochs=30, batch_size=5, validation_split=0.1, verbose=1, callbacks=[checkpoint_callback])

# Loss下降图
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
