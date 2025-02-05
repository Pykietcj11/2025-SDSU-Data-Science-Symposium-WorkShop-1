import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load Llama 3.2 model via Ollama
llm = Ollama(model="llama3.2")

# Load FAISS index
retrieved_faiss = FAISS.load_local(
    "/Users/jaylindyson/Library/CloudStorage/OneDrive-Personal/docs_for_rag/faiss_index",
    SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

#ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
System Role:
You are an expert AI assistant specializing in AI research and technology. You analyze news articles and provide factual answers based only on the given documents.

Instructions:

Use only the provided document text to answer questions about DeepSeek.
Ignore unrelated text such as advertisements, web links, or generic website content.
If a question cannot be answered based on the provided documents, state that the information is unavailable rather than making assumptions.
Summarize key insights concisely while preserving technical accuracy.
If multiple perspectives or comparisons exist within the articles, present a balanced summary.

Response Format:

Direct Answer: Provide a clear and precise answer based on the documents.
Supporting Evidence: Reference relevant details from the articles.
Clarification (if needed): If data is conflicting or unclear, mention it.

Document Text:
{context}

Question: {question}
""")

# --- Sidebar for options ---
st.sidebar.title("Options")

# Input field for user question
st.title("LLM-Powered AI Assistant")
question = st.text_input("Enter Your Question:")

# Perform FAISS search and generate response when button is clicked
if st.button("Get Answer") and question:
    # Retrieve the most relevant chunks from FAISS
    docs = retrieved_faiss.similarity_search(question, k=10)

    # Extract text content from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare the prompt and call Llama 3.2
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    st.subheader("AI Response")
    st.write(response)

