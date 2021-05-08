import tensorflow as tf

from Dataset import labels_cols

criteria_scores = {
    'pigment_network_numeric': tf.constant([0., 0., 2.], dtype=tf.float32),
    'blue_whitish_veil_numeric': tf.constant([0., 2.], dtype=tf.float32),
    'vascular_structures_numeric': tf.constant([0., 0., 2.], dtype=tf.float32),
    'pigmentation_numeric': tf.constant([0., 0., 1.], dtype=tf.float32),
    'streaks_numeric': tf.constant([0., 0., 1.], dtype=tf.float32),
    'dots_and_globules_numeric': tf.constant([0., 0., 1.], dtype=tf.float32),
    'regression_structures_numeric': tf.constant([0, 1.], dtype=tf.float32)
}


def diagnosis(concepts,threshold):
    score = {}
    task_score = {}
    diag = []
    for idx in range(len(concepts)):
        score[idx] = 0
        for idx1, t in enumerate(labels_cols):
            task_score[idx1] = criteria_scores[t][concepts.iloc[idx][t].item()]
            score[idx] += task_score[idx1]
        diag.append(1 if score[idx] >= threshold else 0)
        task_score = {}
    return diag
