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

# Streamlit page config
st.set_page_config(
    page_title="AI SQL Data Analyst Agent",
    layout="wide"
)

st.title("🚀 AI SQL Data Analyst Agent")
st.write("Upload CSV → Convert to SQL → Ask Questions → Get Insights")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Show preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Create SQLite connection
        conn = sqlite3.connect("data.db")

        # Store dataframe into SQL table
        df.to_sql(
            "sales_data",
            conn,
            if_exists="replace",
            index=False
        )

        st.success("CSV converted to SQL database successfully!")

        # User question
        user_question = st.text_input(
            "Ask your question in plain English"
        )

        if st.button("Generate Insights"):

            if not user_question:
                st.warning("Please enter a question")
                st.stop()

            if not groq_api:
                st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit secrets.")
                st.stop()

            # Get schema
            schema = ", ".join(df.columns)

            # Prompt
            prompt = f"""
            You are an expert SQL query generator.

            Database Table Name: sales_data

            Available Columns:
            {schema}

            Convert the user's natural language question into a valid SQLite SQL query.

            Rules:
            1. Use table name sales_data
            2. If column names contain spaces, wrap them in double quotes
            3. Return ONLY SQL query
            4. No explanation
            5. No markdown formatting

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

                # Remove markdown formatting if present
                sql_query = sql_query.replace("```sql", "")
                sql_query = sql_query.replace("```", "")
                sql_query = sql_query.strip()

                # Display SQL query
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")

                # Execute query
                result = pd.read_sql_query(
                    sql_query,
                    conn
                )

                # Show result
                st.subheader("Query Result")
                st.dataframe(result)

                # Visualization
                if len(result.columns) >= 2:
                    st.subheader("Visualization")

                    fig, ax = plt.subplots(figsize=(8, 5))

                    result.plot(
                        x=result.columns[0],
                        y=result.columns[1],
                        kind="bar",
                        ax=ax
                    )

                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)

            except Exception as llm_error:
                st.error(f"Groq/SQL Error: {llm_error}")

        conn.close()

    except Exception as file_error:
        st.error(f"File Error: {file_error}")