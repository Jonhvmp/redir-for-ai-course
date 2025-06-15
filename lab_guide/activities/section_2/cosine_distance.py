import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

# Definindo o modelo
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Frases de exemplo
sentences = [
    "Eu gosto de programar.",
    "Eu amo escrever código.",
    "A natureza é linda.",
    "A programação é divertida.",
    "Essa é uma pessoa muito feliz",
    "Esse é um cachorro feliz",
    "Hoje é um dia ensolarado"
]

# query para comparação
sentence = "Eu gosto de programar."

# Gerando embeddings para as frases
embeddings = model.encode(sentences)

# vector embedding da query
query_embedding = model.encode(sentence)

# calculando a distância do cosseno
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

# rodando a comparação
print('Query: Eu gosto de programar' )
for e, s in zip(embeddings, sentences):
  print(s, " -> Similaridade = ", cosine_distance(e, query_embedding))
