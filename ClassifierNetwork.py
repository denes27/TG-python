import tensorflow as tf
import tensorflow.keras as keras


N_NEURONS_L1 = 1280
N_NEURONS_L2 = 1080
N_NEURONS_L3 = 862
INPUT_SHAPE = tf.TensorShape([256, 862])
RESHAPE_LAYER_SHAPE = (256, 862)
LEARNING_RATE = 0.0001
CLIP_NORM = 0.001


def generateModel():
    # Initializing Device Specification
    device_spec = tf.DeviceSpec(job="localhost", replica=0, device_type="GPU")
    name = ''
    # Specifying the device
    with tf.device(device_spec):
        # build network topology
        model = keras.Sequential([

            # input layer
            # keras.layers.Flatten(input_shape=(mfccc.shape[1], mfccc.shape[2])),

            # 1st dense layer
            keras.layers.Dense(N_NEURONS_L1, input_shape=INPUT_SHAPE, activation='relu'),

            # 2nd dense layer
            keras.layers.Dense(N_NEURONS_L2, activation='relu'),

            # 3rd dense layer
            keras.layers.Dense(N_NEURONS_L3, activation='relu'),

            # output layer
            # keras.layers.Dense(13, activation='softmax'),
            keras.layers.Reshape(RESHAPE_LAYER_SHAPE)
        ])
        for layer in model.layers:
            print(layer.output_shape)

        # compile model
        optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
        model.compile(optimizer=optimiser,
                      loss="mse",
                      metrics=['accuracy'])

        model.summary()

        return model
