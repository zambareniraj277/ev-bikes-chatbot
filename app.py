import streamlit as st
from langchain_groq.chat_models import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="EV Chatbot - Examiner Interface",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENV & KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Load Vectorstore (already built & persisted) ---
PERSIST_DIR = "data/output/chroma"  # Fixed path to match your actual ChromaDB location
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if vectorstore exists
if not Path(PERSIST_DIR).exists():
    st.error(f"❌ Vectorstore not found at {PERSIST_DIR}")
    st.info("Please ensure the ChromaDB has been created using the RAG pipeline first.")
    st.stop()

try:
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.success("✅ Vectorstore loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading vectorstore: {str(e)}")
    st.stop()

# --- Initialize Groq LLM ---
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.info("Please set your GROQ_API_KEY environment variable")
    st.stop()

try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.0
    )
    st.success("✅ Groq LLM initialized successfully!")
except Exception as e:
    st.error(f"❌ Error initializing Groq LLM: {str(e)}")
    st.stop()

# --- Build Conversational RAG Chain ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# --- Header ---
st.title("🧪 EV Industry Market Data Chatbot - Examiner Interface")
st.markdown("**Test the RAG-powered chatbot with pre-processed EV documents**")

# --- Sidebar with Testing Information ---
with st.sidebar:
    st.header("📋 Testing Information")
    
    st.subheader("📂 Pre-processed Documents")
    st.markdown("""
    - **EV_Specifications.pdf** - Technical specifications
    - **EV_Policies.pdf** - Government policies and regulations  
    - **EV_Reports.csv** - Market data and reports
    - **PostgreSQL** - Database integration (coming soon)
    """)
    
    st.subheader("🔧 System Status")
    st.success("✅ Vectorstore: Loaded")
    st.success("✅ LLM: Groq DeepSeek")
    st.success("✅ RAG Chain: Ready")
    
    st.subheader("📊 Test Metrics")
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "successful_queries" not in st.session_state:
        st.session_state.successful_queries = 0
    
    st.metric("Total Queries", st.session_state.total_queries)
    st.metric("Successful Queries", st.session_state.successful_queries)
    
    st.subheader("🧹 Actions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.successful_queries = 0
        st.rerun()
    
    if st.button("Reset Memory"):
        memory.clear()
        st.success("Memory cleared!")

# --- Main Chat Interface ---
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role, content = msg["role"], msg["content"]
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)

# User input
user_query = st.chat_input("🧪 Ask your test question here...")

if user_query:
    # Show user message
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.total_queries += 1
    
    # Query RAG chain with error handling
    try:
        with st.spinner("🤔 Processing your question..."):
            result = qa_chain({"question": user_query})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
        
        # Show assistant response
        st.chat_message("assistant").write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.successful_queries += 1
        
        # Show source documents if available
        if source_docs:
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.markdown("---")
        
        # Success indicator
        st.success("✅ Response generated successfully!")
        
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        st.chat_message("assistant").write(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.error(f"Error: {str(e)}")

# --- Testing Guidelines ---
with st.expander("🧪 Testing Guidelines"):
    st.markdown("""
    ### Suggested Test Questions:
    
    **Basic Functionality:**
    - "What are the main features of electric vehicles?"
    - "Tell me about EV charging infrastructure"
    
    **Document Retrieval:**
    - "What policies exist for EV adoption?"
    - "What are the technical specifications for EVs?"
    
    **Data Analysis:**
    - "What are the current EV market trends?"
    - "Compare different EV models"
    
    **Edge Cases:**
    - "What is the future of hydrogen fuel cells?"
    - "How do EVs impact the power grid?"
    
    ### Testing Checklist:
    - ✅ Response relevance to question
    - ✅ Source document accuracy
    - ✅ Conversation memory retention
    - ✅ Error handling
    - ✅ Response generation speed
    """)

# --- Footer ---
st.markdown("---")
st.caption("🧪 Examiner Interface - EV Industry Market Data Chatbot | Powered by LangChain + Groq + ChromaDB")
