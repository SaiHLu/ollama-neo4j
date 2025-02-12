import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_llm = ChatOllama(model="llama3.2")

# region Method1
# instructions = SystemMessage(
#     content="""
#       You are a surfer dude, having a conversation about the surf conditions on the beach.
#       Respond using surfer slang.
#     """
# )

# question = HumanMessage(content="What's the weather like?")

# response = chat_llm.invoke([instructions, question])

# print(response.content)
# endregion

# region Method2
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
#         ),
#         ("human", "{question}"),
#     ]
# )

# chat_chain = prompt | chat_llm | StrOutputParser()
# response = chat_chain.invoke({"question": "What's the weather like?"})
# print(response)
# endregion

# region Method3
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        ("human", "{question}"),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

response = chat_chain.invoke(
    {"context": current_weather, "question": "What's the weather like at Fistral?"}
)

print(response)
# endregion
