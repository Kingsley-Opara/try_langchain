from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import UpstashVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_KEY_API")

UPSTASH_VECTOR_REST_URL = os.environ.get("UPSTASH_VECTOR_REST_URL")
                                         
UPSTASH_VECTOR_REST_TOKEN = os.environ.get("UPSTASH_VECTOR_REST_TOKEN")




if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


store = UpstashVectorStore(
    embedding=embeddings,
    index_url = UPSTASH_VECTOR_REST_URL,
    index_token = UPSTASH_VECTOR_REST_TOKEN 
)



retriever = store.as_retriever(
    search_type = "similarity",
    search_kwargs = {'k':2}
)


# retriever.invoke("what is the city name after trees")


LLM_CONFIG = {
    "model": "gemini-1.5-pro",
    "google_api_key": GEMINI_API_KEY

}

llm = GoogleGenerativeAI(**LLM_CONFIG)


message = '''
Answer this question using the provided context only.

{question}

Context:
{context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "if the user is greeting then respond to the greeting"),
        ("human", message)
    ]
)


runnable = RunnableParallel(
    passed = RunnablePassthrough(),
    modified = lambda x: x["num"] + 1
)

parser = StrOutputParser()

def get_chain():
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser
    return chain


