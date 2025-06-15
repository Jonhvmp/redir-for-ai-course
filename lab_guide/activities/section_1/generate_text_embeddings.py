from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
text = "Este é um documento técnico, ele descreve o chip de som SID do Commodore 64"
embedding = model.encode(text)

print(embedding[:10])
