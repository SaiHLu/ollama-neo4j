from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain.schema import Document
from langchain.chains import RetrievalQA

# Prepare some documents
# documents = [
#     Document(
#         metadata={"title": "Independence Day"},
#         page_content="A movie where aliens land and attack earth.",
#     ),
#     Document(
#         metadata={"title": "The Day After Tomorrow"},
#         page_content="A movie where the world is hit by a series of extreme weather events.",
#     ),
# ]

chat_llm = ChatOllama(model="llama3.2")

embedding_provider = OllamaEmbeddings(model="llama3.2")

graph = Neo4jGraph(
    url="bolt://44.203.226.202:7687",
    username="neo4j",
    password="shortages-increases-forests",
)

# Programmatic way to create a vector index
# new_vector = Neo4jVector.from_documents(
#     documents=documents,
#     embedding=embedding_provider,
#     graph=graph,
#     index_name="myVectorIndex",
#     embedding_node_property="embedding",
#     text_node_property="plot",
#     create_id_index=True,
#     node_label="Chunk",
# )

movie_plot_vector = Neo4jVector(
    embedding=embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True,
)

response = plot_retriever.invoke({"query": "A romance movie"})

print(response)

# result = movie_plot_vector.similarity_search("A romance movie.")

# for doc in result:
#     print(doc.metadata["title"], "-", doc.page_content)
