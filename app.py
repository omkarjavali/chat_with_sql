from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import os
from datetime import datetime

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  # URL encode the password to handle special characters
  import urllib.parse
  encoded_password = urllib.parse.quote_plus(password)
  db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  # Get the SQL query
  query = sql_chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
  
  # Execute the query and get the response
  response_data = db.run(query)
  
  # Try to convert to DataFrame for display if possible
  try:
    if response_data and response_data.strip() and response_data != "No results returned.":
      # If response contains data, extract it and create a DataFrame
      df = pd.read_sql_query(query, db._engine)
    else:
      df = None
  except Exception as e:
    st.error(f"Error converting results to table: {str(e)}")
    df = None
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    Do not introduce yourself or use phrases like "I'm a SQL assistant" - just provide a direct, professional analysis of the data.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  chain = (
    prompt 
    | llm
    | StrOutputParser()
  )
  
  # Get the AI's natural language response
  response_text = chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
    "query": query,
    "schema": db.get_table_info(),
    "response": response_data
  })
  
  return query, df, response_text

# App starts here
load_dotenv()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! Ask me anything about your database."),
    ]

# Set page configuration
st.set_page_config(
    page_title="Chat with MySQL", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Base styles and dark mode support */
    .main {
        background-color: #121212;
        color: #e0e0e0;
    }

    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #121212;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }

    .chat-message.user {
        background-color: #253141;
        color: #e0e0e0;
    }

    .chat-message.bot {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #81c2fe;
        font-weight: 600;
    }

    h1 {
        font-size: 2rem;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1f1f1f;
        padding-top: 2rem;
    }

    .stSidebar .block-container {
        padding-top: 1rem;
    }

    /* Connection status indicators */
    .status-connected {
        background-color: #2e4d30;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #55a766;
        color: #a4e0ad;
        font-weight: bold;
    }

    .status-disconnected {
        background-color: #4d3c2e; 
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #bb9e68;
        color: #e6ce9c;
        font-weight: bold;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1565c0;
        color: white;
        font-weight: 500;
        border-radius: 0.3rem;
        padding: 0.75rem 1rem;
        border: none;
        width: 100%;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #1976d2;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
        background-color: #2c2c2c;
        color: #e0e0e0;
    }

    /* Info boxes */
    .info-box {
        background-color: #1e3242;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1976d2;
        color: #e0e0e0;
    }

    /* SQL Query display */
    .sql-query {
        background-color: #1e1e1e;
        color: #dcdcdc;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        overflow-x: auto;
    }

    /* Results table */
    .results-table {
        border: 1px solid #444;
        border-radius: 0.3rem;
        overflow: hidden;
        background-color: #1a1a1a;
    }

    /* Example questions section */
    .example-questions {
        background-color: #29303b;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #1976d2;
        color: #e0e0e0;
    }

    .example-questions h4 {
        margin-top: 0;
        color: #81c2fe;
    }

    .example-questions ul {
        margin-bottom: 0;
        color: #e0e0e0;
    }

    /* Fix for dark mode specific elements */
    [data-testid="stAppViewContainer"] {
        background-color: #121212;
    }

    [data-testid="stHeader"] {
        background-color: #121212;
    }

    /* Chat input field */
    [data-testid="stChatInput"] {
        background-color: #2c2c2c;
        border: 1px solid #444;
        border-radius: 20px;
        padding: 8px 16px;
        color: #e0e0e0;
    }
    
    /* Fix for chat input box: ensure text is visible */
    [data-testid="stChatInput"] > div {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stChatInput"] input {
        color: #e0e0e0 !important;
    }

    /* Custom styling for metrics */
    [data-testid="stMetric"] {
        background-color: #2c2c2c;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #e0e0e0;
    }

    [data-testid="stMetricLabel"] {
        color: #bbb;
    }

    [data-testid="stMetricValue"] {
        color: #81c2fe;
    }
    
    /* DataFrames styling */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    .dataframe th {
        background-color: #2c2c2c !important;
        color: #81c2fe !important;
    }
    
    .dataframe td {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: #1e1e1e !important;
    }
    
    .st-ae {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
    }
    
    /* Expander styling */
    .streamlit-expander {
        background-color: #1a1a1a !important;
        border: 1px solid #444 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2c2c2c !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Fix for dark text on dark background in various components */
    div[data-baseweb="select"] {
        background-color: #2c2c2c !important;
    }
    
    div[data-baseweb="base-input"] {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
    }
    
    .stSelectbox label {
        color: #e0e0e0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1976d2 !important;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn.cdnlogo.com/logos/m/10/mysql.svg", width=80)
with col2:
    st.title("Chat with MySQL Database")
    st.markdown("<p style='color: #bbb; margin-top: -10px;'>A natural language interface to query your MySQL database</p>", unsafe_allow_html=True)

# Add a horizontal line for visual separation
st.markdown("<hr style='margin: 15px 0; border: 0; height: 1px; background: #444;'>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #81c2fe;'>✨ Connection Settings</h2>", unsafe_allow_html=True)
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Connection status indicator
    if "db" in st.session_state:
        st.markdown("""
        <div style="background-color:#2e4d30; padding:10px; border-radius:5px; border-left:5px solid #55a766;">
            <p style="color:#a4e0ad; margin:0; font-weight:bold;">✅ Connected to database</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#4d3c2e; padding:10px; border-radius:5px; border-left:5px solid #bb9e68;">
            <p style="color:#e6ce9c; margin:0; font-weight:bold;">⚠️ Not connected</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create expandable section for database settings
    with st.expander("Database Connection", expanded=True):
        # Database connection settings with icons
        st.markdown("##### MySQL Server")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("🖥️ Host", value="localhost", key="Host")
        with col2:
            st.text_input("🔌 Port", value="3306", key="Port")
        
        st.markdown("##### Authentication")
        st.text_input("👤 Username", value="root", key="User")
        st.text_input("🔒 Password", type="password", value="", key="Password")
        st.text_input("📁 Database", value="example", key="Database")
    
    # GROQ API Key input in its own section
    with st.expander("API Settings", expanded=True):
        st.markdown("##### API Configuration")
        api_key = st.text_input("🔑 GROQ API Key", type="password", key="groq_api_key", 
                               help="Required for natural language processing")
    
    # Connection button with enhanced styling
    if st.button("🚀 Connect to Database", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please provide your GROQ API key")
        else:
            # Set the API key for Groq
            os.environ["GROQ_API_KEY"] = api_key
            
            with st.spinner("Establishing connection..."):
                try:
                    db = init_database(
                        st.session_state["User"],
                        st.session_state["Password"],
                        st.session_state["Host"],
                        st.session_state["Port"],
                        st.session_state["Database"]
                    )
                    st.session_state.db = db
                    st.session_state.connected_time = datetime.now().strftime("%H:%M:%S")
                    st.success("🎉 Successfully connected to database!")
                    # Removed st.balloons() for more professional appearance
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
    
    # Add some info at the bottom of sidebar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#1e3242; padding:15px; border-radius:5px; font-size:0.8em;">
        <p style="margin:0; color:#bbb;">
            <strong>💡 Tip:</strong> Ask questions in plain English about your data, and the AI will translate them to SQL.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
# Display chat container with styling
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Check if we have a database connection
if "db" in st.session_state:
    # Display database connection info
    st.markdown(f"""
    <div class="info-box">
        <p><strong>🔗 Connected to:</strong> {st.session_state.Database} @ {st.session_state.Host}:{st.session_state.Port} (connected at {st.session_state.get('connected_time', 'unknown')})</p>
    </div>
    """, unsafe_allow_html=True)

# Create a container for the chat messages with custom styling
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(f"<div class='chat-message bot'>{message.content}</div>", unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👤"):
                st.markdown(f"<div class='chat-message user'>{message.content}</div>", unsafe_allow_html=True)

# Add query examples if no chat history beyond the welcome message
if len(st.session_state.chat_history) <= 1:
    st.markdown("""
    <div style="background-color: #29303b; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #1976d2;">
        <h4 style="margin-top: 0; color: #81c2fe;">Example Questions</h4>
        <ul style="margin-bottom: 0;">
            <li>Show me all tables in the database</li>
            <li>What are the top 5 customers by total orders?</li>
            <li>What's the average order value by month?</li>
            <li>Which products have the highest inventory level?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Add a styled chat input
user_query = st.chat_input("Ask a question about your data...")
if user_query is not None and user_query.strip() != "":
    # Make sure we have an active database connection
    if "db" not in st.session_state:
        st.error("Please connect to a database first!")
        st.stop()
        
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            try:
                # Get the SQL query, dataframe result, and AI response
                query, df, response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                
                # Display the SQL query with nice formatting
                st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>📝 Generated SQL Query</h4>", unsafe_allow_html=True)
                st.code(query, language="sql")
                
                # Display the result table if available
                if df is not None and not df.empty:
                    st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>📊 Query Results</h4>", unsafe_allow_html=True)
                    
                    # Create a styled dataframe with enhanced visual appeal
                    st.markdown("<div class='results-table'>", unsafe_allow_html=True)
                    st.dataframe(
                        df,
                        column_config={col: st.column_config.Column(col, help=f"Column: {col}") for col in df.columns},
                        use_container_width=True,
                        height=min(400, 35 + 35 * len(df)),
                        hide_index=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add stats about the results
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Rows", f"{len(df)}")
                    with cols[1]:
                        st.metric("Columns", f"{len(df.columns)}")
                    with cols[2]:
                        st.metric("Query Time", f"{datetime.now().strftime('%H:%M:%S')}")
                
                # Display the AI's response
                st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>💬 Explanation</h4>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);'>{response}</div>", unsafe_allow_html=True)
                
                # Add only the text response to chat history
                full_response = response
                st.session_state.chat_history.append(AIMessage(content=full_response))
                
            except Exception as e:
                error_msg = f"Error processing your query: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))