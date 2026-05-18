"""
==========================================
Terminal Commands to Setup and Run:
==========================================
1. To install the required libraries, run this command in the Terminal:
   pip install streamlit langchain langchain-community sentence-transformers faiss-cpu ollama

2. To download the local model (make sure the Ollama app is running first):
   ollama pull gemma

3. To run the application:
   streamlit run app.py
==========================================
"""

# Imports and runtime guard (use dynamic imports to reduce editor unresolved-import warnings)
import sys
import importlib

# Prefer normal import for Streamlit at runtime, but import dynamically so
# static analysis tools don't always flag missing packages in editors.
try:
    st = importlib.import_module("streamlit")
except Exception:
    print("Missing dependency: streamlit. Install with: pip install streamlit")
    raise

# Dynamically import optional/third-party modules used by RAG. If any are
# unavailable, we'll surface a clear error when trying to initialize the RAG.
def _dynamic_import(name, attr=None):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None

Ollama = _dynamic_import("langchain_community.llms", "Ollama") or _dynamic_import("ollama", "Ollama")
HuggingFaceEmbeddings = _dynamic_import("langchain.embeddings", "HuggingFaceEmbeddings")
FAISS = _dynamic_import("langchain.vectorstores", "FAISS")
Document = _dynamic_import("langchain.schema", "Document")
PromptTemplate = _dynamic_import("langchain.prompts", "PromptTemplate")
RunnablePassthrough = _dynamic_import("langchain.runnables", "RunnablePassthrough")
StrOutputParser = _dynamic_import("langchain.output_parsers", "StrOutputParser")

# If the file is executed directly with `python app.py`, show a helpful
# message and exit. The app must be run with Streamlit: `streamlit run app.py`.
if __name__ == "__main__":
    try:
        mod = importlib.import_module("streamlit.runtime.scriptrunner.script_run_context")
        get_script_run_ctx = getattr(mod, "get_script_run_ctx", None)
        if get_script_run_ctx is None or get_script_run_ctx() is None:
            raise RuntimeError("not running under streamlit")
    except Exception:
        print("Please run this app with: streamlit run app.py")
        sys.exit(0)

# ==========================================
# 1. UI Settings and Medical Disclaimer
# ==========================================
st.set_page_config(page_title="UnPanic | AI First Aid", page_icon="🧠", layout="centered")

# UX Improvement: Hide default Streamlit menus to reduce visual clutter
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Legal and Medical Disclaimer
st.warning("⚠️ **Important Notice:** UnPanic is a psychological first-aid support tool and is NOT a substitute for a doctor or therapist. If you are in immediate danger, please go to the nearest emergency room.")

st.title("🧠 UnPanic: Psychological First Aid")
st.info("🔒 **Privacy Preserved:** This app runs 100% locally on your device. No data is sent externally.")

# ==========================================
# 2. RAG Setup (Vector DB)
# ==========================================
@st.cache_resource
def setup_rag_system():
    # Ensure required integrations are available
    missing = []
    if Ollama is None:
        missing.append("Ollama (langchain_community.llms or ollama)")
    if HuggingFaceEmbeddings is None:
        missing.append("HuggingFaceEmbeddings (langchain.embeddings)")
    if FAISS is None:
        missing.append("FAISS (langchain.vectorstores)")
    if Document is None:
        missing.append("Document (langchain.schema)")
    if PromptTemplate is None:
        missing.append("PromptTemplate (langchain.prompts)")
    if RunnablePassthrough is None:
        missing.append("RunnablePassthrough (langchain.runnables)")
    if StrOutputParser is None:
        missing.append("StrOutputParser (langchain.output_parsers)")
    if missing:
        raise ImportError("Missing required packages: " + ", ".join(missing))
    # Trauma and panic-focused techniques
    therapies = [
        "Box Breathing Technique: Inhale for 4 seconds, hold your breath for 4 seconds, exhale for 4 seconds, wait for 4 seconds. Repeat this 4 times to calm the nervous system.",
        "5-4-3-2-1 Sensory Grounding: Look around and name: 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, 1 thing you can taste. This anchors you back to reality.",
        "Physical Grounding: Place your feet firmly on the floor. Feel the weight of your body on the chair or bed. Tell yourself: 'I am here, I am safe in this present moment'.",
        "Container Exercise: Imagine a strong box with a sturdy lock. Temporarily place your scary or overwhelming thoughts inside it and lock the box. You can open it later when you are with a professional.",
        "For tremors or muscle tension: Tense your shoulder and arm muscles as tightly as you can for 5 seconds, then release them suddenly. Notice the contrast between tension and relaxation."
    ]
    docs = [Document(page_content=t) for t in therapies]
    
    with st.spinner('Preparing the safe psychological support environment... (This may take a moment on the first run)'):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
    
    llm = Ollama(model="gemma", temperature=0.0)  # keep deterministic outputs to reduce hallucination risk
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    prompt = PromptTemplate.from_template("""
    You are "UnPanic", an empathetic and humane psychological first-aid assistant.
    
    Your mission: Provide immediate reassurance and ONLY ONE practical technique from the context below.
    
    Strict Rules:
    1. Do not diagnose, do not mention medications, and do not over-apologize. Be calm and grounding.
    2. Use simple, direct, and comforting English.
    3. If you cannot find a suitable technique in the context, use "Slow deep breathing" as the default action. Do not invent techniques.
    4. Do not ignore the chat history, but rely heavily on the therapeutic context for your practical advice.
    
    Available Therapeutic Context:
    {context}
    
    Recent Chat History:
    {chat_history}
    
    Current User Message:
    {question}
    
    Your Response (short, reassuring, and containing exactly one practical step):
    """)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

try:
    rag_chain = setup_rag_system()
except Exception as e:
    st.error(f"Failed to load the model. Please ensure Ollama is running in the background. Error: {e}")
    st.stop()

# ==========================================
# 3. Session & Memory Management
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_recent_history():
    # Only keep the last 2 messages to avoid overwhelming the context window
    recent_msgs = st.session_state.messages[-2:]
    if not recent_msgs:
        return "No previous history."
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. Advanced Emergency Trigger & Interaction
# ==========================================
user_input = st.chat_input("How are you feeling right now? (I am here to listen)...")

def check_emergency(text):
    # Expanded list covering direct threats and common metaphorical expressions of severe distress
    keywords = [
        "suicide", "die", "kill myself", "end my life", "hurt myself", 
        "end it all", "no point in living", "want to sleep forever", 
        "tired of breathing", "don't want to wake up", "better off dead"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if check_emergency(user_input):
        response = (
            "🚨 **CRITICAL MESSAGE:**\n\n"
            "We hear how much pain you are in right now, but please do not make a permanent decision in a temporary moment of distress.\n\n"
            "**You are not alone.** Please contact a professional or someone you trust immediately:\n"
            "- **Emergency Services:** Please dial your local emergency number (e.g., 911, 112, 999) or go to the nearest emergency room.\n"
            "- **Crisis Text Line:** Text HOME to 741741 (US/Canada) or 85258 (UK).\n\n"
            "For right now, just try to focus on taking one breath at a time. Inhale... Exhale..."
        )
        with st.chat_message("assistant"):
            st.error(response) 
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    else:
        with st.chat_message("assistant"):
            # A calming text placeholder instead of an anxiety-inducing spinning wheel
            placeholder = st.empty()
            placeholder.markdown("🌿 *Listening to you and finding the best way to support you...*")
            
            try:
                history = get_recent_history()
                response = rag_chain.invoke(
                    {"question": user_input, "chat_history": history}
                )
                
                # Clear placeholder and show the actual response
                placeholder.empty()
                st.markdown(response)
                
            except Exception as e:
                placeholder.empty()
                st.error("A technical error occurred. Please check your local Ollama connection.")
                response = f"Sorry, I encountered a technical issue. ({e})"
    
        st.session_state.messages.append({"role": "assistant", "content": response})