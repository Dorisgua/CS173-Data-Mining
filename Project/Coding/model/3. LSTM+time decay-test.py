import numpy as np
import tensorflow as tf
import pickle
model = tf.keras.models.load_model('LSTM_decay_model.keras', compile=False)


def custom_loss(y_true, y_pred):
    absolute_diff = tf.abs(y_true - y_pred)
    nonzero_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    zero_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    loss = tf.reduce_sum(tf.multiply(absolute_diff, nonzero_mask))
    loss += 0.1*tf.reduce_sum(tf.multiply(absolute_diff, zero_mask))
    return loss

model.compile(optimizer='adam', loss=custom_loss)


X = np.load('../embedding/X_decay.npy')
y_grid = np.load('../embedding/y_grid.npy')
t = 0.05
correct = 0
incorrect = 0
iteration = 200
with open("../data/grid_to_cluster.pkl", "rb") as f:
    grid_to_cluster = pickle.load(f)

for i, (x, y_true) in enumerate(zip(X, y_grid)):
    if iteration == 0:
        break
    y_true = tuple(map(int, y_true[0].strip('()').split(', ')))
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    happen = np.where(y_pred > t)[1]

    grids = [grid_to_cluster[key] for key in happen]
    grids = [point for sublist in grids for point in sublist]
    if any(np.array_equal(y_true, np.array(grid)) for grid in grids):
        print(f"Prediction for sample {i + 1}: Correct")
        correct += 1
    else:
        print(f"Prediction for sample {i + 1}: Incorrect")
        incorrect += 1
    iteration -= 1

print(correct, incorrect)

