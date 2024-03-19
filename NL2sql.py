import os
from sqlalchemy.dialects.postgresql import JSON

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from flask import Flask, request, jsonify
import psycopg2 

os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__7a2e134f3afc4daeaed6fcbe2fb11fd2"



db = SQLDatabase.from_uri("postgresql://postgres:123@localhost/flask_database")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
generate_query = create_sql_query_chain(llm, db)  
execute_query = QuerySQLDataBaseTool(db=db)

app = Flask(__name__)
    
def execute_question(question):
    try:
        output = dict()
        prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
            Question: {question}
            SQL Query: {query}
            SQL Table: {table}
            Answer: """
        )         
        chain = (
            RunnablePassthrough.assign(query=generate_query).assign(
                table=itemgetter("query") | execute_query
            )    
        )        
        info = chain.invoke({"question": question})
        output["info"] = info     
        rephraser_chain = prompt | llm | StrOutputParser()
        rephrased = rephraser_chain.invoke({"question": question, "query": info["query"], "table": info["table"]})
        output["information"] = rephrased
    except Exception as e:
        output = {"error": str(e)}  
    return output

@app.route('/answer_sql', methods=['POST'])
def answer_sql():
    try:
        data = request.json
        user_question = data.get('question')
    
        if not user_question:
                return jsonify({"error": "Invalid input: Question is missing"}), 400
            
        output = execute_question(user_question)
        return jsonify(output)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

if __name__ == '__main__':
    app.run(debug=True)