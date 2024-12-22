from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
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
    
    # Generate the SQL query
    sql_query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
        "schema": db.get_table_info(),
    })
    
    # Execute the SQL query on the database
    sql_response = db.run(sql_query)
    
    # Prepare a natural language response using streaming
    response_placeholder = st.empty()
    response_stream = ""
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = template.format(
        schema=db.get_table_info(),
        chat_history=chat_history,
        query=sql_query,
        question=user_query,
        response=sql_response,
    )
    
    llm = ChatOpenAI(model="gpt-4", temperature=0, stream=True)  # Enable streaming
    
    with st.spinner("Processing..."):
        for chunk in llm.stream(prompt):
            # Extract the text content from the chunk
            response_stream += chunk.content if hasattr(chunk, "content") else str(chunk)
            response_placeholder.markdown(response_stream)
    
    return response_stream.strip()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("💬 Chat with Your Database")

with st.sidebar:
    st.subheader("🔧 Settings")
    st.write("Connect to your MySQL database to start querying.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("🔌 Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("✅ Connected to database!")

st.markdown("---")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type your query here...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    # AI response
    with st.chat_message("AI"):
        response_stream = get_response(user_query, st.session_state.db, st.session_state.chat_history)
    
    # Update chat history after streaming is complete
    st.session_state.chat_history.append(AIMessage(content=response_stream))
    
    # Feedback section
    feedback_placeholder = st.empty()
    if feedback_placeholder.radio("Was this response helpful?", ["Yes", "No"], key=f"feedback_{len(st.session_state.chat_history)}"):
        st.success("Thank you for your feedback!")
