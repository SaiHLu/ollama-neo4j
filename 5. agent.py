from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph, Neo4jChatMessageHistory
from langchain.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import YouTubeSearchTool
from uuid import uuid4

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

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
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | chat_llm | StrOutputParser()

youtube = YouTubeSearchTool()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)


tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(chat_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while (q := input("> ")) != "exit":

    response = chat_agent.invoke(
        {"input": q},
        {"configurable": {"session_id": SESSION_ID}},
    )

    print(response["output"])
