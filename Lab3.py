import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Part A: Streaming chatbot with memory
# ---------------------------------------------------------------------------

st.title("Lab 3 - Streaming Chatbot with Memory")
st.write(
    "Ask me anything. I'll answer simply, then ask if you want more info, "
    "say **yes** to dig deeper, or **no** to move on to something new."
)
st.caption("🧠 Memory: this chatbot remembers only the last 2 exchanges (your last 2 questions and my last 2 answers).")

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Session state holds the full conversation history for display.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Part B: Conversation buffer — keep only the last 2 exchanges
# (last 2 user messages + the LLM's responses to those messages)
# ---------------------------------------------------------------------------

def build_buffer() -> list[dict]:
    """
    Trim the full conversation history down to what gets sent to the LLM:
    only the last 2 user/assistant pairs (at most 4 messages). The system
    prompt (Part C) is added back in separately, after this runs, so
    trimming can never accidentally drop it.
    """
    history = st.session_state.messages
    return history[-4:]


# ---------------------------------------------------------------------------
# Part C: Refine the chatbot
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly assistant that explains things so clearly that a "
    "10-year-old could understand them. Follow this exact pattern:\n"
    "1. When the user asks a new question, answer it simply, then ask "
    "'Do you want more info?'\n"
    "2. If the user replies with something like 'yes', give one more useful "
    "layer of detail on that same topic, then ask 'Do you want more info?' again.\n"
    "3. If the user replies with something like 'no', stop adding detail on that "
    "topic and instead ask what else you can help them with.\n"
    "Always keep answers short, friendly, and easy for a kid to follow."
)

# ---------------------------------------------------------------------------
# Chat input + streaming response
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask me something, or reply yes/no to my last question!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    
    messages_to_send = [{"role": "system", "content": SYSTEM_PROMPT}] + build_buffer()

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})