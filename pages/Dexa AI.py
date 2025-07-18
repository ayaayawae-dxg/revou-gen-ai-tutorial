import streamlit as st
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
import agents.FAQ_DEXA as FAQ_DEXA
import uuid

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Milvus(
    embedding_function=embedding_model,
    connection_args={
        "uri": "http://192.168.76.22:19531",
        "token": "root:Milvus",
        "db_name": "digicamp_ai_miniproject",
    },
    collection_name="bona_dxg_smart",
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong",
    drop_old=False,
    enable_dynamic_field=True,
    auto_id=True,
    primary_field="id",
)

st.title("Dexa AI")

# Initialize session state for messages and chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())

def save_current_chat():
    """Save the current chat to history"""
    if st.session_state.messages:
        # Create title from first user message
        first_user_msg = next((msg for msg in st.session_state.messages if msg["role"] == "user"), None)
        if first_user_msg:
            chat_title = first_user_msg["content"][:30] + "..." if len(first_user_msg["content"]) > 30 else first_user_msg["content"]
            
            # Update existing chat or create new one
            existing_chat = next((c for c in st.session_state.chat_history if c["id"] == st.session_state.current_chat_id), None)
            if existing_chat:
                existing_chat["messages"] = st.session_state.messages.copy()
                existing_chat["title"] = chat_title
            else:
                st.session_state.chat_history.append({
                    "id": st.session_state.current_chat_id,
                    "title": chat_title,
                    "messages": st.session_state.messages.copy()
                })

# Sidebar for chat history
with st.sidebar:
    st.header("Chat History")
    
    # New chat button
    if st.button("➕ New Chat", use_container_width=True):
        # Save current chat to history if it has messages
        if st.session_state.messages:
            chat_title = st.session_state.messages[0]["content"][:30] + "..." if len(st.session_state.messages[0]["content"]) > 30 else st.session_state.messages[0]["content"]
            st.session_state.chat_history.append({
                "id": st.session_state.current_chat_id,
                "title": chat_title,
                "messages": st.session_state.messages.copy()
            })
        
        # Start new chat
        st.session_state.messages = []
        st.session_state.current_chat_id = str(uuid.uuid4())
        # st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Previous Chats")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # Show if this is the current chat
            is_current = chat["id"] == st.session_state.current_chat_id
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Add visual indicator for current chat
                button_text = f"{'🔵' if is_current else '💬'} {chat['title']}"
                if st.button(button_text, key=f"chat_{chat['id']}", use_container_width=True):
                    if not is_current:  # Only switch if not already current
                        # Save current chat before switching
                        save_current_chat()
                        
                        # Load selected chat
                        st.session_state.messages = chat["messages"].copy()
                        st.session_state.current_chat_id = chat["id"]
                        # st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
                    st.session_state.chat_history = [c for c in st.session_state.chat_history if c["id"] != chat["id"]]
                    # If we're deleting the current chat, start a new one
                    if chat["id"] == st.session_state.current_chat_id:
                        st.session_state.messages = []
                        st.session_state.current_chat_id = str(uuid.uuid4())
                    # st.rerun()
            
            # Show message count
            msg_count = len(chat["messages"])
            st.caption(f"💬 {msg_count} messages")
    
    # Clear all history button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.chat_history = []
            # st.rerun()

# Main chat area header
col1, col2 = st.columns([4, 1])
with col1:
    if st.session_state.messages:
        # Show current chat info
        msg_count = len(st.session_state.messages)
        st.caption(f"💬 Current chat • {msg_count} messages")
    else:
        st.caption("💬 Start a new conversation")

with col2:
    if st.button("🗑️ Clear", use_container_width=True, help="Clear current chat"):
        st.session_state.messages = []
        # st.rerun()

st.divider()

# Display chat messages in the state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accepting user input
prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Auto-save after user input
    save_current_chat()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        status_placeholder.status(label="Process Start")
        state = "Process Start"
        final_answer = ""
        
        for chunk, metadata in FAQ_DEXA.graph.stream({"messages":HumanMessage(content=prompt)}, stream_mode="messages"):
            current_node = metadata['langgraph_node']
            
            if current_node != state:
                status_placeholder.status(label=current_node)
                state = current_node
                print(f"Node changed to: {current_node}")
            
            # Only accumulate content from the generate node (final answer)
            if current_node == "generate":
                final_answer += chunk.content
                answer_placeholder.markdown(final_answer)
        
        status_placeholder.status(label="Complete", state='complete')
    
        # Append to session state only once, after the entire graph is complete
    st.session_state.messages.append({"role":"assistant", "content": final_answer})
    
    # Auto-save the current chat after each interaction
    save_current_chat()