import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser

llm = OllamaLLM(model="llama3.2", temperature=0)

template = PromptTemplate(
    template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""",
    input_variables=["fruit"],
)

llm_chain = template | llm | SimpleJsonOutputParser()

response = llm_chain.invoke({"fruit": "apple"})

print(response)
