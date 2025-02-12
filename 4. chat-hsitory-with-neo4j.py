from langchain_neo4j import Neo4jGraph, Neo4jChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableWithMessageHistory
from uuid import uuid4

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

# Local Neo4j
# graph = Neo4jGraph(
#     url="bolt://localhost:7687",
#     username="neo4j",
#     password="password",
# )

chat_llm = ChatOllama(model="llama3.2")

graph = Neo4jGraph(
    url="bolt://44.203.226.202:7687",
    username="neo4j",
    password="shortages-increases-forests",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Bells", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while (question := input("> ")) != "exit":
    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
        },
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print(response)

# result = graph.query(
#     """
#   MATCH (m:Movie{title: 'Toy Story'})
#   RETURN m.title, m.plot, m.poster
# """
# )

# print(graph.schema)
