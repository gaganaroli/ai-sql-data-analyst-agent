import streamlit as st
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api = os.getenv("GROQ_API_KEY")

# Streamlit title
st.title("AI SQL Data Analyst Agent (CSV → SQL → Insights)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Store CSV data into SQLite database
    conn = sqlite3.connect("data.db")
    df.to_sql(
        "sales_data",
        conn,
        if_exists="replace",
        index=False
    )

    st.success("CSV converted to SQL database successfully!")

    # Natural language input
    user_question = st.text_input(
        "Ask your question in plain English"
    )

    if st.button("Generate Insights"):

        # Get schema
        schema = ", ".join(df.columns)

        # Prompt for SQL generation
        prompt = f"""
        You are an expert SQL query generator.

        Database Table Name: sales_data

        Available Columns:
        {schema}

        Convert the user's natural language question into a valid SQLite SQL query.

        Rules:
        1. Use table name: sales_data
        2. If column names contain spaces, wrap them in double quotes
        3. Return ONLY SQL query
        4. Do not add explanation

        User Question:
        {user_question}
        """

        try:
            # Groq LLM
            llm = ChatGroq(
                groq_api_key=groq_api,
                model_name="llama-3.3-70b-versatile"
            )

            response = llm.invoke(prompt)

            sql_query = response.content.strip()

            # Show generated SQL
            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")

            # Execute SQL query
            result = pd.read_sql_query(sql_query, conn)

            # Show results
            st.subheader("Query Result")
            st.dataframe(result)

            # Visualization
            if len(result.columns) >= 2:
                st.subheader("Visualization")

                result.plot(
                    x=result.columns[0],
                    y=result.columns[1],
                    kind="bar",
                    figsize=(8,5)
                )

                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(plt)

        except Exception as e:
            st.error(f"Error: {e}")