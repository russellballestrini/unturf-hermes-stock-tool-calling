import streamlit as st
from streamlit_chat import message
import sqlite3
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
import plotly
import json
from datetime import datetime
from openai import OpenAI
import pandas as pd
import logging

# --------------------------- Configuration ---------------------------

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Configure Hermes API
API_KEY = "choose-any-value"  # Replace with your actual API key or use environment variables
BASE_URL = "https://hermes.ai.unturf.com/v1"
MODEL = "NousResearch/Hermes-3-Llama-3.1-8B"

# Initialize OpenAI client
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Define tools
def fetch_stock_price(ticker: str):
    try:
        logging.debug(f"Fetching stock price for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        price = stock.info.get('currentPrice') or stock.history(period="1d")['Close'].iloc[-1]
        logging.debug(f"Fetched price for {ticker}: ${price:.2f}")
        return {"type": "text", "content": f"The current price of {ticker.upper()} is ${price:.2f}"}
    except Exception as e:
        logging.error(f"Error fetching stock price for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching stock price for {ticker}: {str(e)}"}

def fetch_stock_chart(ticker: str, period: str = "1mo", interval: str = "1d"):
    try:
        logging.debug(f"Fetching stock chart for ticker: {ticker}, period: {period}, interval: {interval}")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError("No historical data found.")
        fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker.upper()} Closing Prices - Last {period}")
        fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
        logging.debug(f"Fetched and created chart for {ticker}")
        # Serialize the Plotly figure to JSON
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return {"type": "plot", "content": fig_json}
    except Exception as e:
        logging.error(f"Error fetching stock chart for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching stock chart for {ticker}: {str(e)}"}

def fetch_multiple_tickers(tickers: list):
    try:
        logging.debug(f"Fetching data for multiple tickers: {tickers}")
        data = yf.download(tickers, period="1mo", interval="1d", group_by='ticker')
        if data.empty:
            raise ValueError("No data found for the provided tickers.")
        logging.debug(f"Fetched data for multiple tickers: {tickers}")
        # Serialize the DataFrame to JSON
        data_json = data.to_json(orient='split')
        return {"type": "dataframe", "content": data_json}
    except Exception as e:
        logging.error(f"Error fetching data for multiple tickers {tickers}: {str(e)}")
        return {"type": "error", "content": f"Error fetching data for multiple tickers {tickers}: {str(e)}"}

def get_financials(ticker: str):
    try:
        logging.debug(f"Fetching financials for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        financials = tck.financials
        if financials.empty:
            raise ValueError("No financial data available.")
        logging.debug(f"Fetched financials for {ticker}")
        # Serialize the DataFrame to JSON
        financials_json = financials.to_json(orient='split')
        return {"type": "dataframe", "content": financials_json}
    except Exception as e:
        logging.error(f"Error fetching financials for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching financials for {ticker}: {str(e)}"}

def get_balance_sheet(ticker: str):
    try:
        logging.debug(f"Fetching balance sheet for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        balance_sheet = tck.balance_sheet
        if balance_sheet.empty:
            raise ValueError("No balance sheet data available.")
        logging.debug(f"Fetched balance sheet for {ticker}")
        # Serialize the DataFrame to JSON
        balance_sheet_json = balance_sheet.to_json(orient='split')
        return {"type": "dataframe", "content": balance_sheet_json}
    except Exception as e:
        logging.error(f"Error fetching balance sheet for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching balance sheet for {ticker}: {str(e)}"}

def get_cash_flow(ticker: str):
    try:
        logging.debug(f"Fetching cash flow for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        cash_flow = tck.cashflow
        if cash_flow.empty:
            raise ValueError("No cash flow data available.")
        logging.debug(f"Fetched cash flow for {ticker}")
        # Serialize the DataFrame to JSON
        cash_flow_json = cash_flow.to_json(orient='split')
        return {"type": "dataframe", "content": cash_flow_json}
    except Exception as e:
        logging.error(f"Error fetching cash flow for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching cash flow for {ticker}: {str(e)}"}

def get_options(ticker: str):
    try:
        logging.debug(f"Fetching options dates for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        options = tck.options
        if not options:
            raise ValueError("No options available.")
        logging.debug(f"Fetched options dates for {ticker}: {options}")
        return {"type": "text", "content": f"Available options dates for {ticker.upper()}: {', '.join(options)}"}
    except Exception as e:
        logging.error(f"Error fetching options for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching options for {ticker}: {str(e)}"}

def get_option_chain(ticker: str, date: str = None):
    try:
        logging.debug(f"Fetching option chain for ticker: {ticker}, date: {date}")
        tck = yf.Ticker(ticker)
        if date is None:
            if not tck.options:
                raise ValueError("No options available.")
            date = tck.options[0]
        option_chain = tck.option_chain(date)
        puts = option_chain.puts
        calls = option_chain.calls
        if puts.empty and calls.empty:
            raise ValueError("No option chain data available.")
        logging.debug(f"Fetched option chain for {ticker} on {date}")
        # Serialize the DataFrames to JSON
        puts_json = puts.to_json(orient='split')
        calls_json = calls.to_json(orient='split')
        return {"type": "tables", "content": {"puts": puts_json, "calls": calls_json, "date": date}}
    except Exception as e:
        logging.error(f"Error fetching option chain for {ticker} on {date}: {str(e)}")
        return {"type": "error", "content": f"Error fetching option chain for {ticker} on {date}: {str(e)}"}

def get_institutional_holders(ticker: str):
    try:
        logging.debug(f"Fetching institutional holders for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        holders = tck.institutional_holders
        if holders is None or holders.empty:
            raise ValueError("No institutional holders data available.")
        logging.debug(f"Fetched institutional holders for {ticker}")
        # Serialize the DataFrame to JSON
        holders_json = holders.to_json(orient='split')
        return {"type": "dataframe", "content": holders_json}
    except Exception as e:
        logging.error(f"Error fetching institutional holders for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching institutional holders for {ticker}: {str(e)}"}

def get_sector_info(ticker: str):
    try:
        logging.debug(f"Fetching sector and industry info for ticker: {ticker}")
        tck = yf.Ticker(ticker)
        info = tck.info
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        logging.debug(f"Fetched sector: {sector}, industry: {industry} for {ticker}")
        return {"type": "text", "content": f"**Sector**: {sector}\n**Industry**: {industry}"}
    except Exception as e:
        logging.error(f"Error fetching sector/industry for {ticker}: {str(e)}")
        return {"type": "error", "content": f"Error fetching sector/industry for {ticker}: {str(e)}"}

def equity_query(query: str):
    try:
        logging.debug(f"Processing equity query: {query}")
        # Placeholder for advanced equity querying logic
        # This can be enhanced based on specific querying requirements
        response = f"Processed equity query: {query}"
        logging.debug(f"Equity query response: {response}")
        return {"type": "text", "content": response}
    except Exception as e:
        logging.error(f"Error processing equity query '{query}': {str(e)}")
        return {"type": "error", "content": f"Error processing equity query '{query}': {str(e)}"}

def screener(criteria: dict):
    try:
        logging.debug(f"Screening stocks with criteria: {criteria}")
        # Placeholder for screening stocks based on criteria
        # This can be enhanced to filter stocks using yfinance or other data sources
        response = f"Screening stocks with criteria: {criteria}"
        logging.debug(f"Screener response: {response}")
        return {"type": "text", "content": response}
    except Exception as e:
        logging.error(f"Error screening stocks with criteria '{criteria}': {str(e)}")
        return {"type": "error", "content": f"Error screening stocks with criteria '{criteria}': {str(e)}"}

# You can expand the TOOLS dictionary with additional tools as needed
TOOLS = {
    "stock_price": {
        "name": "Fetch Stock Price",
        "description": "Fetches the current price of a given stock ticker.",
        "function": fetch_stock_price
    },
    "stock_chart": {
        "name": "Fetch Stock Chart",
        "description": "Fetches and plots the closing price chart for a given stock ticker over a specified period.",
        "function": fetch_stock_chart
    },
    "multiple_tickers": {
        "name": "Fetch Multiple Tickers",
        "description": "Fetches historical data for multiple stock tickers simultaneously.",
        "function": fetch_multiple_tickers
    },
    "financials": {
        "name": "Get Financials",
        "description": "Retrieves financial statements for a given stock ticker.",
        "function": get_financials
    },
    "balance_sheet": {
        "name": "Get Balance Sheet",
        "description": "Retrieves the balance sheet for a given stock ticker.",
        "function": get_balance_sheet
    },
    "cash_flow": {
        "name": "Get Cash Flow",
        "description": "Retrieves the cash flow statement for a given stock ticker.",
        "function": get_cash_flow
    },
    "options": {
        "name": "Get Options",
        "description": "Retrieves available options dates for a given stock ticker.",
        "function": get_options
    },
    "option_chain": {
        "name": "Get Option Chain",
        "description": "Retrieves the option chain (puts and calls) for a given stock ticker on a specified date.",
        "function": get_option_chain
    },
    "institutional_holders": {
        "name": "Get Institutional Holders",
        "description": "Retrieves institutional holders information for a given stock ticker.",
        "function": get_institutional_holders
    },
    "sector_info": {
        "name": "Get Sector and Industry Info",
        "description": "Retrieves sector and industry information for a given stock ticker.",
        "function": get_sector_info
    },
    "equity_query": {
        "name": "Equity Query",
        "description": "Processes advanced equity queries based on user input.",
        "function": equity_query
    },
    "screener": {
        "name": "Stock Screener",
        "description": "Screens stocks based on various user-defined criteria.",
        "function": screener
    }
}

# --------------------------- Database Setup ---------------------------

# Establish connection with SQLite database
conn = sqlite3.connect('chat_app.db', check_same_thread=False)
c = conn.cursor()

# Create 'rooms' table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS rooms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
''')

# Create 'messages' table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id INTEGER,
        sender TEXT,
        content TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(room_id) REFERENCES rooms(id)
    )
''')

conn.commit()

# --------------------------- Helper Functions ---------------------------

def get_room_id(room_name: str):
    logging.debug(f"Retrieving room ID for room: {room_name}")
    c.execute("SELECT id FROM rooms WHERE name = ?", (room_name,))
    result = c.fetchone()
    if result:
        logging.debug(f"Found existing room ID: {result[0]} for room: {room_name}")
        return result[0]
    else:
        c.execute("INSERT INTO rooms (name) VALUES (?)", (room_name,))
        conn.commit()
        new_id = c.lastrowid
        logging.debug(f"Created new room: {room_name} with ID: {new_id}")
        return new_id

def get_messages(room_id: int):
    logging.debug(f"Fetching messages for room ID: {room_id}")
    c.execute("SELECT sender, content, timestamp FROM messages WHERE room_id = ? ORDER BY timestamp", (room_id,))
    messages = c.fetchall()
    logging.debug(f"Fetched {len(messages)} messages for room ID: {room_id}")
    return messages

def save_message(room_id: int, sender: str, content: str):
    logging.debug(f"Saving message to room ID: {room_id}, sender: {sender}, content: {content[:50]}...")
    c.execute("INSERT INTO messages (room_id, sender, content) VALUES (?, ?, ?)", (room_id, sender, content))
    conn.commit()
    logging.debug(f"Message saved to room ID: {room_id}")

def analyze_tools_via_ai(user_message: str):
    try:
        logging.debug(f"Analyzing tools via AI for user message: {user_message}")
        # Prepare the list of tools for the system prompt
        tools_description = "\n".join([f"- **{key}**: {tool['description']}" for key, tool in TOOLS.items()])
        system_prompt = f"""
You are an AI assistant tasked with determining which tools to call based on the user's message. Below is a list of available tools:

{tools_description}

When the user asks a question, decide which tools are relevant and should be called to fulfill the request. Use as few tools as necessary and do not call tools greedily. Return a JSON array of tool names to be called. If no tools are needed, return an empty array.

User Message: "{user_message}"

Response Format:
["tool_name1", "tool_name2"]
even if it's empty: []
"""
        logging.debug(f"Tool determination system prompt: {system_prompt}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            temperature=0.0,
            max_tokens=100
        )
        tool_output = response.choices[0].message.content.strip()
        logging.debug(f"AI tool determination response: {tool_output}")
        # Parse the JSON array
        tool_names = json.loads(tool_output)
        logging.debug(f"Tools to be called: {tool_names}")
        return tool_names
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error during tool analysis via AI: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error during tool analysis via AI: {str(e)}")
        return []

def render_tool_response(tool_response: dict):
    try:
        if tool_response["type"] == "text":
            st.markdown(f"**Tool:** {tool_response['content']}")
        elif tool_response["type"] == "plot":
            # Deserialize the Plotly figure from JSON
            fig = json.loads(tool_response["content"])
            st.plotly_chart(fig, use_container_width=True)
        elif tool_response["type"] == "dataframe":
            # Deserialize the DataFrame from JSON
            df = pd.read_json(tool_response["content"], orient='split')
            st.dataframe(df)
        elif tool_response["type"] == "tables":
            st.subheader(f"Option Chain for {tool_response['content']['date']}")
            st.write("**Calls:**")
            calls_df = pd.read_json(tool_response['content']['calls'], orient='split')
            st.dataframe(calls_df)
            st.write("**Puts:**")
            puts_df = pd.read_json(tool_response['content']['puts'], orient='split')
            st.dataframe(puts_df)
        elif tool_response["type"] == "error":
            st.error(tool_response["content"])
        else:
            st.warning(f"Unknown tool response type: {tool_response['type']}")
    except Exception as e:
        logging.error(f"Error rendering tool response: {str(e)}")
        st.error(f"Error rendering tool response: {str(e)}")

def get_chat_system_prompt(messages: list, tool_responses: list):
    try:
        logging.debug("Generating chat system prompt with context history and tool responses.")
        # Combine tool responses into JSON
        tools_output = json.dumps(tool_responses, default=str)
        # Prepare chat history
        chat_history = ""
        for msg in messages:
            if msg['sender'] == "user":
                chat_history += f"User: {msg['content']}\n"
            elif msg['sender'] == "assistant":
                chat_history += f"Assistant: {msg['content']}\n"
            elif msg['sender'] == "tool":
                chat_history += f"Tool: {msg['content']}\n"
        system_prompt = f"""
You are a helpful assistant. Here is the conversation history:

{chat_history}

If tools were called, here are their responses:
{tools_output}

Respond to the user's latest message based on the above context.
"""
        logging.debug(f"Chat system prompt: {system_prompt}")
        return system_prompt
    except Exception as e:
        logging.error(f"Error generating chat system prompt: {str(e)}")
        return "You are a helpful assistant."

def process_user_message(room_id: int, user_message: str):
    logging.debug(f"Processing user message in room ID: {room_id}")
    save_message(room_id, "user", user_message)
    
    # Determine which tools to call via AI
    triggered_tools = analyze_tools_via_ai(user_message)
    tool_responses = []
    
    if triggered_tools:
        logging.debug(f"Tools triggered: {triggered_tools}")
        for tool_name in triggered_tools:
            tool = TOOLS.get(tool_name)
            if tool:
                # Extract additional parameters if necessary
                # For simplicity, assume only ticker is needed unless it's equity_query or screener
                ticker = extract_ticker(user_message)
                if not ticker and tool_name not in ["equity_query", "screener"]:
                    # For tools that require a ticker, if no ticker found, add an error
                    error_msg = f"Could not extract ticker from message: {user_message}"
                    logging.warning(error_msg)
                    tool_responses.append({"type": "error", "content": error_msg})
                    save_message(room_id, "tool", json.dumps({"type": "error", "content": error_msg}))
                    continue

                try:
                    logging.debug(f"Executing tool: {tool_name} with ticker: {ticker}")
                    if tool_name == "option_chain":
                        # Extract date if mentioned
                        date = extract_date(user_message)
                        response = tool["function"](ticker, date)
                    elif tool_name == "multiple_tickers":
                        tickers = extract_multiple_tickers(user_message)
                        response = tool["function"](tickers)
                    elif tool_name in ["equity_query", "screener"]:
                        response = tool["function"](user_message)
                    else:
                        response = tool["function"](ticker)
                    tool_responses.append(response)
                    save_message(room_id, "tool", json.dumps(response))
                except Exception as e:
                    error_content = f"Error executing tool '{tool_name}': {str(e)}"
                    logging.error(error_content)
                    tool_responses.append({"type": "error", "content": error_content})
                    save_message(room_id, "tool", json.dumps({"type": "error", "content": error_content}))
            else:
                logging.warning(f"Tool not found: {tool_name}")
                error_content = f"Tool '{tool_name}' is not available."
                tool_responses.append({"type": "error", "content": error_content})
                save_message(room_id, "tool", json.dumps({"type": "error", "content": error_content}))
    else:
        logging.debug("No tools were triggered.")
    
    # Prepare messages for Hermes
    messages = []
    # Fetch all previous messages
    all_msgs = get_messages(room_id)
    for sender, content, _ in all_msgs:
        messages.append({"sender": sender, "content": content})
    
    # Get chat system prompt
    chat_system_prompt = get_chat_system_prompt(messages, tool_responses)
    
    # Call Hermes API for chat response
    try:
        logging.debug("Calling Hermes API for chat response.")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": chat_system_prompt},
            ],
            temperature=0.5,
            max_tokens=300
        )
        assistant_reply = response.choices[0].message.content.strip()
        logging.debug(f"Hermes API response: {assistant_reply}")
    except Exception as e:
        logging.error(f"Error communicating with Hermes API: {str(e)}")
        assistant_reply = f"Error communicating with Hermes API: {str(e)}"
    
    # Save assistant reply
    save_message(room_id, "assistant", assistant_reply)
    
    return assistant_reply, tool_responses

def extract_ticker(user_message: str):
    try:
        logging.debug(f"Extracting ticker from user message: {user_message}")
        # Simple extraction: find the first word that looks like a ticker (e.g., uppercase letters, possibly with $)
        words = user_message.upper().replace('$', '').split()
        for word in words:
            if word.isalpha() and 1 <= len(word) <= 5:  # Assuming tickers are up to 5 letters
                logging.debug(f"Ticker extracted: {word}")
                return word
        logging.debug("No valid ticker found in user message.")
        return None
    except Exception as e:
        logging.error(f"Error extracting ticker: {str(e)}")
        return None

def extract_multiple_tickers(user_message: str):
    try:
        logging.debug(f"Extracting multiple tickers from user message: {user_message}")
        words = user_message.upper().replace('$', '').split()
        potential_tickers = [word for word in words if word.isalpha() and 1 <= len(word) <= 5]
        tickers = list(set(potential_tickers))  # Remove duplicates
        logging.debug(f"Multiple tickers extracted: {tickers}")
        return tickers
    except Exception as e:
        logging.error(f"Error extracting multiple tickers: {str(e)}")
        return []

def extract_date(user_message: str):
    try:
        logging.debug(f"Extracting date from user message: {user_message}")
        words = user_message.lower().split()
        if "on" in words:
            idx = words.index("on")
            if idx + 1 < len(words):
                date = words[idx + 1]
                logging.debug(f"Date extracted: {date}")
                return date
        logging.debug("No date found in user message.")
        return None
    except Exception as e:
        logging.error(f"Error extracting date: {str(e)}")
        return None

# --------------------------- Streamlit Interface ---------------------------

st.set_page_config(page_title="Hermes Chat Enhanced", layout="wide")

st.title("🗨️ Hermes Chat Enhanced Application with Debugging")

# Sidebar for room management
st.sidebar.header("📁 Chat Rooms")
room_names = [row[0] for row in c.execute("SELECT name FROM rooms").fetchall()]
if room_names:
    selected_room = st.sidebar.selectbox("Select a room", room_names)
else:
    selected_room = None
    st.sidebar.warning("No rooms available. Create a new room.")

# Initialize session state for current_room
if 'current_room' not in st.session_state:
    st.session_state.current_room = selected_room

# Update current_room based on selection
if selected_room and st.session_state.current_room != selected_room:
    st.session_state.current_room = selected_room

if st.session_state.current_room:
    room_id = get_room_id(st.session_state.current_room)
    
    # Display chat messages
    messages = get_messages(room_id)
    st.subheader(f"💬 {st.session_state.current_room}")
    for sender, content, timestamp in messages:
        if sender == "user":
            message(content, is_user=True, key=f"{timestamp}-user")
        elif sender == "assistant":
            message(content, is_user=False, key=f"{timestamp}-assistant")
        elif sender == "tool":
            # Attempt to parse JSON
            try:
                tool_content = json.loads(content)
                render_tool_response(tool_content)
            except json.JSONDecodeError:
                st.markdown(f"**Tool:** {content}")
    
    # Input area
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", key="input", height=100)
        submit = st.form_submit_button(label="Send", on_click=None)
    
    if submit and user_input.strip() != "":
        with st.spinner("Processing..."):
            assistant_reply, tool_responses = process_user_message(room_id, user_input)
        st.rerun()

# Room creation
st.sidebar.subheader("➕ Create New Room")
new_room = st.sidebar.text_input("Room Name")
if st.sidebar.button("Create"):
    if new_room.strip() != "":
        try:
            new_room_id = get_room_id(new_room.strip())
            st.session_state.current_room = new_room.strip()  # Switch to the new room
            st.sidebar.success(f"✅ Room '{new_room}' created and switched to it.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.sidebar.error("❌ Room with this name already exists.")
    else:
        st.sidebar.error("⚠️ Room name cannot be empty.")

# Enable Ctrl+Enter to submit
st.markdown(
    """
    <script>
    const textarea = window.parent.document.querySelector('textarea');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                const buttons = window.parent.document.querySelectorAll('button');
                buttons.forEach(button => {
                    if (button.textContent === 'Send') {
                        button.click();
                    }
                });
            }
        });
    }
    </script>
    """,
    unsafe_allow_html=True
)
