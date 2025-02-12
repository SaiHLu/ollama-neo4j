import ollama

text = "Toy Story is a movie about..."
response = ollama.embeddings(model="llama3.2", prompt=text)
print(len(response.embedding))  # Should print 3072
