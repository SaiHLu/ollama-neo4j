from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

llm = ChatOllama(model="llama3.2")

graph = Neo4jGraph(
    url="bolt://3.239.225.142:7687",
    username="neo4j",
    password="brook-flag-pleasure",
)

# The schema will be automatically generated from the graph database and passed to the LLM. The question will be the user’s question.
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE, input_variables=["schema", "question"]
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True,
)

result = cypher_chain.invoke({"query": "What is the plot of the movie Toy Story?"})

print(result)
