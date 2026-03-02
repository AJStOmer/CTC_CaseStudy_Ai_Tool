#imports
import streamlit as st
import pandas as pd
from groq import Groq
import duckdb
import os



#load Groq llm
#client = Groq(api_key=st.secrets["GROQ_API_KEY"])
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


# -----------------------------
# Professional Header
# -----------------------------
st.markdown("""
<div style='background-color:#1f4e79; padding:18px; border-radius:6px; margin-bottom:20px;'>
    <h1 style='color:white; text-align:center; margin:0;'>AI Analytics Query Tool</h1>
    <p style='color:#d9e2ef; text-align:center; margin:0; font-size:14px;'>Powered by Groq LLM + DuckDB SQL Execution</p>
</div>
""", unsafe_allow_html=True)

#load csv
df = pd.read_csv("CTC_CaseStudy_Dataset.csv")

#build a schema summary to help the llm generate correct SQL queries
schema_summary = []
for col in df.columns:
    dtype = str(df[col].dtype)
    uniqueCount = df[col].nunique()
    repeated = uniqueCount < len(df)
    is_numeric = pd.api.types.is_numeric_dtype(df[col])

    desc = f"{col}: dtype={dtype}, "
    desc += "repeated" if repeated else "mostly unique"
    if is_numeric:
        desc += ", numeric (aggregatable)"
    else:
        desc += ", non-numeric"

    schema_summary.append(desc)

schema_summary = "\n".join(schema_summary)

st.success("Dataset successfully loaded.")

# -----------------------------
# About Section
# -----------------------------
st.markdown("""
### About This Tool
<div style='background-color:#f5f7fa; padding:15px; border-radius:8px; line-height:1.6;'>
This AI tool was created to support the analytics portion of this case study.

It works by sending your question to an LLM hosted by Groq, along with the dataset’s column names and data types.  
The model generates a SQL query, which is executed against the dataset to compute the answer.

<b>Note:</b> Responses may not always be perfect. For best results, ask clear and specific questions.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Example Questions Box
# -----------------------------
st.markdown("""
### Example Questions
<div style='background-color:#eef2f7; padding:15px; border-radius:8px;'>
• What were the number of Units Sold in 2023?<br>
• What were the number of Units Sold in 2024?<br>
• What was the revenue in 2023?<br>
• What was the revenue in 2024?<br>
• What Region has the Highest Profit?<br>
• Which Quarter had the Highest number of Units Sold?<br>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

#User input for data question
user_query = st.text_input("Enter your analytics question:")

if user_query:

    #Groq prompt
    prompt = f"""
You are an expert Data Analyst. You will write a SQL query that computes the answer.

Dataset schema (with datatype + repeated/unique info):
{schema_summary}

User question: {user_query}

Rules:
- Never use backticks of any kind.
- Never wrap SQL in code fences.
- Do NOT write any explanations. 
- Do NOT write any suggestions.
- Output ONLY raw SQL.
- Do NOT use window functions (no RANK, ROW_NUMBER, DENSE_RANK, etc.).
- If a column is marked "repeated", treat it as a grouping key.
- If a column is numeric and repeated, aggregate it (SUM, AVG, MAX, MIN) before ordering.
- For "top", "bottom", "highest", "lowest", or "rank" style questions, use simple GROUP BY + ORDER BY + LIMIT.
- Use only the table named df.
"""

    #recieve groq response
    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    #clean SQL response
    sql = llm_response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    st.markdown("### Generated SQL")
    st.code(sql, language="sql")

    #use duckdb to run the SQL query and print the results
    try:
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(sql).fetchall()

        st.markdown("### Query Result")

        if len(result) == 1 and len(result[0]) == 1:
            st.write(result[0][0])
        else:
            st.dataframe(pd.DataFrame(result))

    except Exception as e:
        st.error(f"Error executing SQL: {e}")
