from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')  

sentences = ["test1", "another sentence"]

embeddings = model.encode(sentences)

print(embeddings.shape)