import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from io import BytesIO
import time

# --- Configuration ---
GEMINI_MODEL_ID = "gemini-1.5-flash-latest"
DEFAULT_K_CONTEXT_CHUNKS = 4
DEFAULT_TEMPERATURE = 0.3

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="🎓 Academic PDF Converser ✨",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"<h1 style='text-align: center; color: #005f73;'>🎓 Academic PDF Converser ✨</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-style: italic;'>Deep dive into research papers with context-aware chat, powered by LangChain & Gemini ({GEMINI_MODEL_ID})!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- API Key Input in Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")
    GEMINI_API_KEY = st.text_input("Enter Google Gemini API Key:", type="password", key="gemini_api_key_sidebar")
    if not GEMINI_API_KEY:
        st.warning("⚠️ Please enter your Gemini API Key to enable chatting.")
    st.markdown("---")
    st.info("This app uses LangChain's `ConversationalRetrievalChain` for a rich Q&A experience.")
    st.markdown("<sub>Advanced Conversational AI</sub>", unsafe_allow_html=True)

# --- Session State Initialization ---
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chat_history_display" not in st.session_state: st.session_state.chat_history_display = []
if "current_pdf_name" not in st.session_state: st.session_state.current_pdf_name = None
if "conversation_object" not in st.session_state: st.session_state.conversation_object = None # Will store {"chain": ..., "memory": ...}


# --- Core LangChain and Helper Functions ---
@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def extract_text_from_pdf(pdf_file_bytes):
    try:
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text if text.strip() else None
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}"); return None

@st.cache_resource
def create_vector_store(_pdf_text_tuple):
    pdf_text = _pdf_text_tuple[0]
    if not pdf_text: return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_text(pdf_text)
        if not chunks: st.warning("⚠️ No text chunks generated."); return None
        embeddings = load_embeddings_model()
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"❌ Error creating vector store: {e}"); return None

# --- Conversation Answering Function (Inspired by your example) ---
def conversation_answering(
    vector_store, 
    question: str, 
    api_key: str,
    k=DEFAULT_K_CONTEXT_CHUNKS, 
    temperature=DEFAULT_TEMPERATURE, 
    conversation_obj=None # Pass existing chain and memory here
):
    if not api_key:
        return {"answer": "Error: API Key is missing.", "source_documents": []}, conversation_obj
    if not vector_store:
        return {"answer": "Error: Vector store (PDF index) is not available.", "source_documents": []}, conversation_obj

    try:
        if conversation_obj is None:
            # Define custom prompt template for the document combination (QA) step
            # This prompt is crucial for guiding the LLM's response style and constraints.
            qa_template = """You are an expert AI research assistant. Your goal is to answer questions based *only* on the provided context from a research paper.
Be precise and technical. If specific details or data are present in the context, include them.
If the context does not contain the information to answer the question, you MUST explicitly state: "The provided context does not contain specific information to answer this question."
Do not make up information or use any external knowledge.

Context from the paper:
---
{context}
---

Question: {question}

Precise Answer based *only* on the context:"""
            QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
                            
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_ID, 
                google_api_key=api_key, 
                temperature=temperature,
                convert_system_message_to_human=True
            )
            
            retriever = vector_store.as_retriever(search_kwargs={'k': k})
            
            # Memory needs to be initialized for each new conversation (i.e., each new PDF)
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                output_key='answer', # Key for the LLM's final answer
                return_messages=True # Store LangChain Message objects
            )
            
            # Prompt for condensing the current question and chat history
            _template_condense = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If the follow up question is already a standalone question or if there is no chat history, just return the question as is.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template_condense)

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                chain_type="stuff", # "stuff" is good for smaller contexts; for very long docs, consider "map_reduce" or "refine"
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT}, # Use our custom QA prompt
                output_key='answer'
            )
            
            conversation_obj = {"chain": chain, "memory": memory}
        
        # Invoke question into the chain
        # The chain's memory automatically handles the chat_history input for the condense_question_prompt
        result = conversation_obj["chain"].invoke({'question': question})
        return result, conversation_obj # result contains 'answer' and 'source_documents'

    except Exception as e:
        st.error(f"Error in conversation answering: {str(e)}")
        return {"answer": f"Error during LangChain processing: {str(e)}", "source_documents": []}, conversation_obj


# --- Main App UI and Logic (No Columns) ---

st.markdown("#### 📤 Upload Your PDF Document")
uploaded_file = st.file_uploader("Drag & Drop or Click to Upload", type=["pdf"], key="pdf_uploader_main", label_visibility="collapsed")

if uploaded_file:
    if st.session_state.current_pdf_name != uploaded_file.name or not st.session_state.vector_store:
        st.session_state.current_pdf_name = uploaded_file.name
        st.session_state.chat_history_display = []
        st.session_state.conversation_object = None # Reset for new PDF
        
        with st.spinner(f"⏳ Analyzing '{uploaded_file.name}'..."):
            pdf_bytes = uploaded_file.getvalue()
            pdf_text = extract_text_from_pdf(pdf_bytes)
            if pdf_text:
                st.session_state.vector_store = create_vector_store(tuple([pdf_text]))
                if st.session_state.vector_store:
                    st.success(f"✅ PDF '{uploaded_file.name}' processed and ready for chat!")
                    # Chain will be initialized on first question if API key is present
                else:
                    st.error("⚠️ Failed to create searchable index from PDF.")
            else:
                st.session_state.vector_store = None # Ensure reset

st.markdown("#### 💬 Chat with Your Document")
chat_container = st.container(height=500)
with chat_container:
    for i, message in enumerate(st.session_state.chat_history_display):
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🔬"):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("source_documents"):
                with st.expander("🔍 View Cited Sources"):
                    for doc_idx, doc in enumerate(message["source_documents"]):
                        page_label = doc.metadata.get('page', f"Chunk {doc_idx+1}") if hasattr(doc, 'metadata') else f"Chunk {doc_idx+1}"
                        st.caption(f"Source ({page_label}):")
                        st.markdown(f"> {doc.page_content.strip()}", unsafe_allow_html=True)
                        if doc_idx < len(message["source_documents"]) - 1: st.markdown("---")

if prompt := st.chat_input(
    "Ask a detailed question about the paper...", 
    disabled=not st.session_state.vector_store or not GEMINI_API_KEY, 
    key="chat_input_main"
):
    st.session_state.chat_history_display.append({"role": "user", "content": prompt})
    with chat_container: # Display user message immediately
        with st.chat_message("user", avatar="🧑‍🎓"): st.markdown(prompt)

    with chat_container: # Process and display assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            thinking_ph = st.empty()
            thinking_text = "thinking..."
            for i in range(len(thinking_text) + 1):
                thinking_ph.markdown(f"{thinking_text[:i]}▌"); time.sleep(0.03)
            thinking_ph.markdown(thinking_text)

            # Call the conversation_answering function
            api_response, updated_conversation_obj = conversation_answering(
                vector_store=st.session_state.vector_store,
                question=prompt,
                api_key=GEMINI_API_KEY,
                conversation_obj=st.session_state.conversation_object
            )
            
            # Persist the updated chain and memory for next turn
            st.session_state.conversation_object = updated_conversation_obj 
            
            answer = api_response.get("answer", "Sorry, an error occurred.")
            source_docs = api_response.get("source_documents", [])
            
            thinking_ph.empty()
            answer_ph = st.empty()
            disp_text = ""
            for char_ans in answer: 
                disp_text += char_ans; answer_ph.markdown(disp_text + "▌"); time.sleep(0.01)
            answer_ph.markdown(answer)

    st.session_state.chat_history_display.append({
        "role": "assistant",
        "content": answer,
        "source_documents": source_docs
    })

if not uploaded_file and not st.session_state.chat_history_display:
    st.info("☝️ Welcome! Please upload a research paper (PDF) to begin your analysis.")
elif not GEMINI_API_KEY:
    st.error("🚫 Please enter your Google Gemini API Key in the sidebar to enable interaction.")
elif uploaded_file and not st.session_state.vector_store:
    st.warning("⏳ PDF processing might have failed or is still in progress. Check for errors or try re-uploading.")
