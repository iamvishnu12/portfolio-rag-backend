import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


def ask_question(query: str):

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant that answers questions about Vishnu Reddy's resume.

Only answer using the provided context.
If the answer is not in the context, say:
"I don't have that information in the resume."
Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content