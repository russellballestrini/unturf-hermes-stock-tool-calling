import streamlit as st
from openai import OpenAI
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
import traceback
import sqlite3
import uuid
from contextlib import contextmanager

# For Streamlit Cloud deployment
import os
from urllib.parse import urlparse

# Database Configuration
def get_database_connection():
    """
    Get database connection based on environment
    For Streamlit Cloud, uses PostgreSQL if DATABASE_URL is set, otherwise SQLite
    """
    if os.getenv('DATABASE_URL'):
        # For Streamlit Cloud with PostgreSQL
        try:
            import psycopg2
            from psycopg2.extras import DictCursor
            
            url = urlparse(os.getenv('DATABASE_URL'))
            conn = psycopg2.connect(
                dbname=url.path[1:],
                user=url.username,
                password=url.password,
                host=url.hostname,
                port=url.port
            )
            return conn
        except ImportError:
            st.error("Please install psycopg2-binary for PostgreSQL support")
            raise
    else:
        # Local SQLite database
        return sqlite3.connect('chat_history.db', check_same_thread=False)

@contextmanager
def get_db_cursor():
    """Context manager for database operations"""
    conn = get_database_connection()
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()

def init_database():
    """Initialize database tables"""
    with get_db_cursor() as cursor:
        # Create chatrooms table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chatrooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (room_id) REFERENCES chatrooms (room_id)
        )
        ''')

def create_chatroom(name: str) -> str:
    """Create a new chatroom and return its ID"""
    room_id = str(uuid.uuid4())
    with get_db_cursor() as cursor:
        cursor.execute(
            'INSERT INTO chatrooms (room_id, name) VALUES (?, ?)',
            (room_id, name)
        )
    return room_id

def get_chatrooms() -> List[Dict[str, Any]]:
    """Get all chatrooms"""
    with get_db_cursor() as cursor:
        cursor.execute('SELECT room_id, name FROM chatrooms ORDER BY created_at DESC')
        return [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]

def get_chat_history(room_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a specific room"""
    with get_db_cursor() as cursor:
        cursor.execute(
            'SELECT role, content FROM messages WHERE room_id = ? ORDER BY created_at',
            (room_id,)
        )
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

def save_message(room_id: str, role: str, content: str):
    """Save a message to the database"""
    message_id = str(uuid.uuid4())
    with get_db_cursor() as cursor:
        cursor.execute(
            'INSERT INTO messages (message_id, room_id, role, content) VALUES (?, ?, ?, ?)',
            (message_id, room_id, role, content)
        )

# System prompts
TOOL_SYSTEM_PROMPT = """
You are a stock market analysis assistant. Use the available tools to provide accurate market information:
1. get_stock_fundamentals(symbol: str): Get company fundamentals
2. get_stock_price(symbol: str): Get current stock price
3. get_company_news(symbol: str): Get latest company news
4. get_stock_dividends(symbol: str): Get dividend information

If the user's request requires market data, respond with a JSON object containing:
{
    "response": "Your natural language response",
    "tool_calls": [
        {"name": "tool_name", "arguments": {"symbol": "TICKER"}}
    ]
}
"""

CHAT_SYSTEM_PROMPT = """
You are a knowledgeable stock market assistant. Engage in conversation about market trends, 
investment strategies, and financial concepts. Use the chat history provided to maintain context 
and provide relevant, informative responses. You can refer to previous discussions and data 
when appropriate.
"""

# [Previous tool functions remain the same]

def handle_user_input(user_input: str, room_id: str) -> None:
    """Process user input and generate response"""
    try:
        # Save user message
        save_message(room_id, "user", user_input)
        
        # Get chat history
        chat_history = get_chat_history(room_id)
        
        # Initialize OpenAI client
        client = OpenAI(base_url="https://hermes.ai.unturf.com/v1", api_key="choose-any-value")
        
        # First, try with tool system prompt
        messages = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT}
        ]
        messages.extend([{"role": msg["role"], "content": msg["content"]} 
                        for msg in chat_history[-5:]])  # Last 5 messages for context
        
        with st.spinner("Analyzing your request..."):
            response = client.chat.completions.create(
                model="NousResearch/Hermes-3-Llama-3.1-8B",
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
        
        llm_output = response.choices[0].message.content
        
        try:
            parsed_response = json.loads(llm_output)
            has_tools = isinstance(parsed_response, dict) and "tool_calls" in parsed_response
        except json.JSONDecodeError:
            has_tools = False
            
        if not has_tools:
            # If no tools needed, use chat system prompt
            messages = [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT}
            ]
            messages.extend([{"role": msg["role"], "content": msg["content"]} 
                           for msg in chat_history])
            
            with st.spinner("Generating response..."):
                response = client.chat.completions.create(
                    model="NousResearch/Hermes-3-Llama-3.1-8B",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
            
            assistant_response = response.choices[0].message.content
            save_message(room_id, "assistant", assistant_response)
            return {"response": assistant_response, "tool_calls": []}
        
        # Handle tool-based response
        save_message(room_id, "assistant", parsed_response["response"])
        return parsed_response
        
    except Exception as e:
        st.error(f"Error processing your request: {str(e)}")
        st.error(traceback.format_exc())
        return None

def main():
    st.title("🤖 Stock Market Analysis Assistant")
    
    # Initialize database
    init_database()
    
    # Chatroom management
    with st.sidebar:
        st.header("Chat Rooms")
        if st.button("Create New Chat Room"):
            room_name = st.text_input("Enter room name:", key="new_room")
            if room_name:
                room_id = create_chatroom(room_name)
                st.session_state.current_room = room_id
                st.success(f"Created room: {room_name}")
        
        rooms = get_chatrooms()
        if rooms:
            selected_room = st.selectbox(
                "Select Chat Room",
                options=[(room["id"], room["name"]) for room in rooms],
                format_func=lambda x: x[1]
            )
            if selected_room:
                st.session_state.current_room = selected_room[0]
    
    # Ensure a room is selected
    if not getattr(st.session_state, 'current_room', None):
        if rooms:
            st.session_state.current_room = rooms[0]["id"]
        else:
            st.session_state.current_room = create_chatroom("General")
    
    # Chat interface
    chat_history = get_chat_history(st.session_state.current_room)
    
    # Display chat history
    for message in chat_history:
        role_icon = "👤" if message["role"] == "user" else "🤖"
        st.markdown(f"{role_icon} **{message['role'].title()}:** {message['content']}")
    
    # User input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("What would you like to know about stocks?", 
                                 key="user_input",
                                 height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            response_data = handle_user_input(user_input, st.session_state.current_room)
            
            if response_data:
                st.markdown("### 🤖 Analysis")
                st.write(response_data["response"])
                
                if tool_calls := response_data.get("tool_calls"):
                    for tool_call in tool_calls:
                        try:
                            tool_name = tool_call["name"]
                            arguments = tool_call["arguments"]
                            
                            if tool_name in TOOLS:
                                with st.spinner(f"Fetching {tool_name.replace('_', ' ')}..."):
                                    result = TOOLS[tool_name](**arguments)
                                    display_tool_result(tool_name, result)
                            else:
                                st.error(f"Unknown tool: {tool_name}")
                        except Exception as e:
                            st.error(f"Error executing {tool_name}: {str(e)}")

if __name__ == "__main__":
    main()
