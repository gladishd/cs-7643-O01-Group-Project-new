(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (f531d50e ✗) python CONVOlUTIONAL_NEURAL_NETWORK_KERAS
python: can't open file 'CONVOlUTIONAL_NEURAL_NETWORK_KERAS': [Errno 2] No such file or directory
(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (f531d50e ✗) python CONVOlUTIONAL_NEURAL_NETWORK_KERAS.py
Filename happy.wav does not match expected format.
Filename sad.wav does not match expected format.
2024-04-22 21:47:18.825815: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2 Pro
2024-04-22 21:47:18.825850: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2024-04-22 21:47:18.825853: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2024-04-22 21:47:18.826095: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:303] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2024-04-22 21:47:18.826377: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:269] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 126, 241, 32)      320

 batch_normalization (Batch  (None, 126, 241, 32)      128
 Normalization)

 max_pooling2d (MaxPooling2  (None, 63, 120, 32)       0
 D)

 conv2d_1 (Conv2D)           (None, 61, 118, 64)       18496

 batch_normalization_1 (Bat  (None, 61, 118, 64)       256
 chNormalization)

 max_pooling2d_1 (MaxPoolin  (None, 30, 59, 64)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 28, 57, 128)       73856

 batch_normalization_2 (Bat  (None, 28, 57, 128)       512
 chNormalization)

 max_pooling2d_2 (MaxPoolin  (None, 14, 28, 128)       0
 g2D)

 flatten (Flatten)           (None, 50176)             0

 dense (Dense)               (None, 64)                3211328

 dropout (Dropout)           (None, 64)                0

 dense_1 (Dense)             (None, 8)                 520

=================================================================
Total params: 3305416 (12.61 MB)
Trainable params: 3304968 (12.61 MB)
Non-trainable params: 448 (1.75 KB)
_________________________________________________________________
Epoch 1/10
2024-04-22 21:47:19.330110: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - ETA: 0s - loss: 5.6527 - accuracy: 0.25002024-04-22 21:47:19.703706: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - 1s 638ms/step - loss: 5.6527 - accuracy: 0.2500 - val_loss: 54.8997 - val_accuracy: 0.0000e+00
Epoch 2/10
1/1 [==============================] - 0s 47ms/step - loss: 21.3980 - accuracy: 0.3750 - val_loss: 52.3831 - val_accuracy: 0.0000e+00
Epoch 3/10
1/1 [==============================] - 0s 42ms/step - loss: 14.2371 - accuracy: 0.6250 - val_loss: 38.8892 - val_accuracy: 0.0000e+00
Epoch 4/10
1/1 [==============================] - 0s 39ms/step - loss: 6.6936 - accuracy: 0.8750 - val_loss: 39.1711 - val_accuracy: 0.5000
Epoch 5/10
1/1 [==============================] - 0s 38ms/step - loss: 14.4420 - accuracy: 0.7500 - val_loss: 38.1529 - val_accuracy: 0.0000e+00
Epoch 6/10
1/1 [==============================] - 0s 37ms/step - loss: 8.2814 - accuracy: 0.6250 - val_loss: 55.3173 - val_accuracy: 0.0000e+00
Epoch 7/10
1/1 [==============================] - 0s 37ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 71.2098 - val_accuracy: 0.0000e+00
Epoch 8/10
1/1 [==============================] - 0s 36ms/step - loss: 10.2476 - accuracy: 0.7500 - val_loss: 88.3299 - val_accuracy: 0.0000e+00
Epoch 9/10
1/1 [==============================] - 0s 37ms/step - loss: 1.3349 - accuracy: 0.8750 - val_loss: 95.0246 - val_accuracy: 0.0000e+00
Epoch 10/10
1/1 [==============================] - 0s 36ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 100.7726 - val_accuracy: 0.0000e+00
Pruned CNN model weights and architecture saved.
Epoch 1/10
1/1 [==============================] - 0s 62ms/step - loss: 2.2735 - accuracy: 0.8750 - val_loss: 91.2566 - val_accuracy: 0.0000e+00
Epoch 2/10
1/1 [==============================] - 0s 39ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 83.3681 - val_accuracy: 0.0000e+00
Epoch 3/10
1/1 [==============================] - 0s 35ms/step - loss: 6.5162 - accuracy: 0.7500 - val_loss: 78.5029 - val_accuracy: 0.0000e+00
Epoch 4/10
1/1 [==============================] - 0s 36ms/step - loss: 4.4703e-07 - accuracy: 1.0000 - val_loss: 74.5232 - val_accuracy: 0.0000e+00
Epoch 5/10
1/1 [==============================] - 0s 36ms/step - loss: 11.8904 - accuracy: 0.8750 - val_loss: 70.9050 - val_accuracy: 0.0000e+00
Epoch 6/10
1/1 [==============================] - 0s 36ms/step - loss: 10.1845 - accuracy: 0.8750 - val_loss: 67.6513 - val_accuracy: 0.0000e+00
Epoch 7/10
1/1 [==============================] - 0s 35ms/step - loss: 0.0000e+00 - accuracy: 1.0000 - val_loss: 67.1174 - val_accuracy: 0.0000e+00
Epoch 8/10
1/1 [==============================] - 0s 35ms/step - loss: 10.0156 - accuracy: 0.8750 - val_loss: 66.3263 - val_accuracy: 0.0000e+00
Epoch 9/10
1/1 [==============================] - 0s 35ms/step - loss: 3.5961 - accuracy: 0.8750 - val_loss: 65.4085 - val_accuracy: 0.0000e+00
Epoch 10/10
1/1 [==============================] - 0s 35ms/step - loss: 6.7119 - accuracy: 0.8750 - val_loss: 62.5072 - val_accuracy: 0.0000e+00
2024-04-22 21:47:24.631587: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - 0s 56ms/step
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/Users/deangladish/miniforge3/envs/tensorflow/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
              precision    recall  f1-score   support

     unknown       0.00      0.00      0.00       1.0
        calm       0.00      0.00      0.00       0.0
       happy       0.00      0.00      0.00       1.0

    accuracy                           0.00       2.0
   macro avg       0.00      0.00      0.00       2.0
weighted avg       0.00      0.00      0.00       2.0

(tensorflow) ~/CS-7643-O01/Group_Project/Data Zenodo (f531d50e ✗)
