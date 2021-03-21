import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3", trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]
embedding_model = tf.keras.Model(text_input, pooled_output)

sentences = tf.constant(["heart doctor"])
print(embedding_model(sentences))
