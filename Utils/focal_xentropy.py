import tensorflow as tf

class FocalCategoricalCrossEntropy(tf.losses.Loss):
    def __init__(self, beta=2., from_logits=True, **kwargs):
        super(FocalCategoricalCrossEntropy, self).__init__(**kwargs)
        self.beta = tf.constant(beta, dtype=tf.float32)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss_uw = tf.losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        w = tf.pow(1 - tf.math.softmax(y_pred, axis=-1), self.beta)
        w *= y_true
        w = tf.reduce_sum(w, axis=-1)
        loss = w * loss_uw
        return tf.reduce_mean(loss, axis=-1)