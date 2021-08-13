from keras.models import model_from_json
import unet_data_load
import common
import numpy as np

subseq = 96
batch_size = 32

# load json and create model
json_file = open('segment_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("segment_model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

read_processed_data = unet_data_load.load_data('Smartforceps', subseq=96)

X = read_processed_data[0]
y = read_processed_data[1]
X_train = read_processed_data[2]
X_test = read_processed_data[3]
y_train = read_processed_data[4]
y_test = read_processed_data[5]
N_FEATURES = read_processed_data[6]
y_map = read_processed_data[7]
act_classes = read_processed_data[8]
class_names = read_processed_data[9]


# after creating test and train data pieces
loaded_model.evaluate(X_test, y_test, batch_size=batch_size)
y_test_resh = y_test.reshape(y_test.shape[0], y_test.shape[2], -1)
y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)

y_pred_raw = loaded_model.predict(X_test, batch_size=batch_size)
y_pred_resh = y_pred_raw.reshape(y_pred_raw.shape[0], y_pred_raw.shape[2], -1)
y_pred_resh_argmax = np.argmax(y_pred_resh, axis=2)
y_pred = y_pred_resh_argmax.reshape(y_pred_resh_argmax.size)
y_pred_prob = y_pred_resh.reshape(y_pred_resh_argmax.size, y_pred_resh.shape[2])
print('prediction results data shape: ', y_pred_prob.shape)

label_index = list(range(1, act_classes + 1))
accuracy, precision, recall, fscore, fw = common.checkAccuracy(labels_test_unary + 1, y_pred + 1, label_index)



