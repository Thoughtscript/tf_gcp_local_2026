import tensorflow as tf


def build(demo: str, x_shape, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=x_shape)
    if demo == "linear":
        out = tf.keras.layers.Dense(1)(inp)
        loss, metrics = "mse", [tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    elif demo == "logistic":
        out = tf.keras.layers.Dense(1, activation="sigmoid")(inp)
        loss, metrics = (
            "binary_crossentropy",
            ["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
    elif demo == "forecast":
        out = tf.keras.layers.Dense(1)(tf.keras.layers.LSTM(32)(inp))
        loss, metrics = "mse", [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    else:
        raise ValueError(demo)
    model = tf.keras.Model(inp, out, name=demo)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model
