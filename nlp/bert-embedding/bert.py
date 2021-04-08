import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics.pairwise import cosine_similarity

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3", trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]

embedding_model = tf.keras.Model(text_input, pooled_output)

query = tf.constant(["brain"])
query_embedding = embedding_model(query).numpy()
print(query_embedding)

documents = [
  { 'id': 1, 'text': "cardiac surgeon" },
  { 'id': 2, 'text': "neuroscientist" },
  { 'id': 3, 'text': "brain surgeon" }
]
print(documents)

document_embeddings = list(
    map(lambda doc:
        { 'id': doc['id'], 'text': embedding_model(tf.constant([doc['text']]).numpy()) },
        documents
    )
)
print(document_embeddings)

cosine_similarities = list(
    map(lambda doc:
        { 'id': doc['id'], 'score': cosine_similarity(query_embedding, doc['text'])[0][0] },
        document_embeddings
    )
)
print(cosine_similarities)

cosine_similarities.sort(key = lambda doc: doc['score'], reverse=True)
print(cosine_similarities)

print("Documents:")
print(documents)

results = list(
    map(lambda score:
        { 'id': score['id'], 'text': list(map(lambda doc: doc['text'], filter(lambda doc: doc['id'] == score['id'], documents)))[0] },
        cosine_similarities
    )
)
print("")
print("Ranked by most similar to search query: '" + query.numpy()[0].decode('ascii') + "'")
print(results)
