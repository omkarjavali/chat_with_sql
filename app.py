from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime
import tempfile
import plotly.express as px
import plotly.graph_objects as go

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    import urllib.parse
    encoded_password = urllib.parse.quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def init_sqlite_database(file_path: str) -> SQLDatabase:
    db_uri = f"sqlite:///{file_path}"
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
    
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def generate_charts(df):
    """Generate appropriate charts based on DataFrame content."""
    if df is None or df.empty:
        return None
    
    charts = []
    
    # Helper function to check if a column is numeric
    def is_numeric(col):
        return pd.api.types.is_numeric_dtype(df[col])
    
    # Helper function to check if a column is categorical
    def is_categorical(col):
        return pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])
    
    # Helper function to check if a column is a date
    def is_datetime(col):
        return pd.api.types.is_datetime64_any_dtype(df[col])
    
    # Identify relevant columns
    categorical_cols = [col for col in df.columns if is_categorical(col)]
    numeric_cols = [col for col in df.columns if is_numeric(col)]
    date_cols = [col for col in df.columns if is_datetime(col)]
    
    # Case 1: Bar chart for categorical vs numeric (e.g., spending by name)
    if categorical_cols and numeric_cols:
        # Prioritize a meaningful categorical column (e.g., name over ID)
        x_col = next((col for col in categorical_cols if "name" in col.lower()), categorical_cols[0])
        # Prioritize a numeric column that indicates spending or total
        y_col = next((col for col in numeric_cols if "total" in col.lower() or "spending" in col.lower() or "amount" in col.lower()), numeric_cols[0])
        
        fig_bar = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
            color=x_col,
            template="plotly_dark"
        )
        fig_bar.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        charts.append(("Bar Chart", fig_bar))
    
    # Case 2: Pie chart for categorical distribution (if aggregation-like data)
    if categorical_cols and numeric_cols and len(df) <= 20:  # Limit pie charts to reasonable sizes
        names_col = categorical_cols[0]
        values_col = next((col for col in numeric_cols if "total" in col.lower() or "spending" in col.lower() or "amount" in col.lower()), numeric_cols[0])
        fig_pie = px.pie(
            df,
            names=names_col,
            values=values_col,
            title=f"Distribution of {values_col.replace('_', ' ').title()} by {names_col.replace('_', ' ').title()}",
            template="plotly_dark"
        )
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        charts.append(("Pie Chart", fig_pie))
    
    # Case 3: Line chart for date/time series
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        y_col = next((col for col in numeric_cols if "total" in col.lower() or "spending" in col.lower() or "amount" in col.lower()), numeric_cols[0])
        fig_line = px.line(
            df.sort_values(date_col),
            x=date_col,
            y=y_col,
            title=f"{y_col.replace('_', ' ').title()} Over Time",
            template="plotly_dark"
        )
        fig_line.update_layout(
            xaxis_title=date_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        charts.append(("Line Chart", fig_line))
    
    return charts if charts else None

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    
    response_data = db.run(query)
    
    try:
        if response_data and response_data.strip() and response_data != "No results returned.":
            df = pd.read_sql_query(query, db._engine)
            # Convert date columns to datetime if applicable
            for col in df.columns:
                if df[col].dtype == "object" and df[col].str.match(r'\d{4}-\d{2}-\d{2}').all():
                    df[col] = pd.to_datetime(df[col])
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
    
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    
    chain = (
        prompt 
        | llm
        | StrOutputParser()
    )
    
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything about your database."),
    ]

st.set_page_config(
    page_title="Chat with Database",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep your existing CSS (unchanged, omitted for brevity)
st.markdown("""
<style>
    /* Same CSS as your original app */
    /* ... */
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn.cdnlogo.com/logos/m/10/mysql.svg", width=80)
with col2:
    st.title("Chat with Database")
    st.markdown("<p style='color: #bbb; margin-top: -10px;'>A natural language interface to query your database</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 15px 0; border: 0; height: 1px; background: #444;'>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #81c2fe;'>✨ Connection Settings</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    
    connection_type = st.radio("Connection Type", ["MySQL Server", "SQLite File"])
    
    if connection_type == "MySQL Server":
        with st.expander("MySQL Connection", expanded=True):
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
    else:
        with st.expander("SQLite File", expanded=True):
            uploaded_file = st.file_uploader("📁 Upload SQLite Database", type=["db", "sqlite", "sqlite3"])
    
    with st.expander("API Settings", expanded=True):
        st.markdown("##### API Configuration")
        api_key = st.text_input("🔑 Mistral API Key", type="password", key="mistral_api_key",
                              help="Required for natural language processing")
    
    if st.button("🚀 Connect to Database", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please provide your Mistral API key")
        else:
            os.environ["MISTRAL_API_KEY"] = api_key
            with st.spinner("Establishing connection..."):
                try:
                    if connection_type == "MySQL Server":
                        db = init_database(
                            st.session_state["User"],
                            st.session_state["Password"],
                            st.session_state["Host"],
                            st.session_state["Port"],
                            st.session_state["Database"]
                        )
                    else:
                        if uploaded_file is None:
                            st.error("Please upload a SQLite database file")
                            st.stop()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        db = init_sqlite_database(tmp_path)
                    
                    st.session_state.db = db
                    st.session_state.connected_time = datetime.now().strftime("%H:%M:%S")
                    st.success("🎉 Successfully connected to database!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#1e3242; padding:15px; border-radius:5px; font-size:0.8em;">
        <p style="margin:0; color:#bbb;">
            <strong>💡 Tip:</strong> Ask questions in plain English about your data, and the AI will translate them to SQL.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

if "db" in st.session_state:
    db_name = st.session_state.get("Database", "SQLite database") if connection_type == "MySQL Server" else "Uploaded SQLite database"
    st.markdown(f"""
    <div class="info-box">
        <p><strong>🔗 Connected to:</strong> {db_name} @ {st.session_state.get('Host', 'local file')}:{st.session_state.get('Port', '')} (connected at {st.session_state.get('connected_time', 'unknown')})</p>
    </div>
    """, unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(f"<div class='chat-message bot'>{message.content}</div>", unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👤"):
                st.markdown(f"<div class='chat-message user'>{message.content}</div>", unsafe_allow_html=True)

if len(st.session_state.chat_history) <= 1:
    st.markdown("""
    <div style="background-color: #29303b; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #1976d2;">
        <h4 style="margin-top: 0; color: #81c2fe;">Example Questions</h4>
        <ul style="margin-bottom: 0;">
            <li>Show me all tables in the database</li>
            <li>What are the top 5 customers by total orders?</li>
            <li>What's the average order value by month?</li>
            <li>Which products have the highest inventory level?</li>
            <li>What is the total revenue by product category?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

user_query = st.chat_input("Ask a question about your data...")
if user_query is not None and user_query.strip() != "":
    if "db" not in st.session_state:
        st.error("Please connect to a database first!")
        st.stop()
        
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            try:
                query, df, response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                
                st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>📝 Generated SQL Query</h4>", unsafe_allow_html=True)
                st.code(query, language="sql")
                
                if df is not None and not df.empty:
                    st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>📊 Query Results</h4>", unsafe_allow_html=True)
                    st.markdown("<div class='results-table'>", unsafe_allow_html=True)
                    st.dataframe(
                        df,
                        column_config={col: st.column_config.Column(col, help=f"Column: {col}") for col in df.columns},
                        use_container_width=True,
                        height=min(400, 35 + 35 * len(df)),
                        hide_index=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Rows", f"{len(df)}")
                    with cols[1]:
                        st.metric("Columns", f"{len(df.columns)}")
                    with cols[2]:
                        st.metric("Query Time", f"{datetime.now().strftime('%H:%M:%S')}")
                    
                    # Generate and display charts
                    charts = generate_charts(df)
                    if charts:
                        st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>📈 Visualizations</h4>", unsafe_allow_html=True)
                        for chart_name, fig in charts:
                            st.markdown(f"<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown(f"<p style='color: #bbb; text-align: center;'>{chart_name}</p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #81c2fe; margin-top: 20px;'>💬 Explanation</h4>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #1a1a1a; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);'>{response}</div>", unsafe_allow_html=True)
                
                full_response = response
                st.session_state.chat_history.append(AIMessage(content=full_response))
                
            except Exception as e:
                error_msg = f"Error processing your query: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))
