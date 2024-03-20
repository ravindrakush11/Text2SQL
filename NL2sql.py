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

os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__7a2e134f3afc4daeaed6fcbe2fb11fd2"

#model name here
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

app = Flask(__name__)

    
def execute_question(question, db):
    try:
        output = dict()
        generate_query = create_sql_query_chain(llm, db)  
        execute_query = QuerySQLDataBaseTool(db=db)

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
        output = {"status": 500, "error": str(e)}  
    return output

def set_db_connection(db_name):
    if db_name == "db1":
        db = SQLDatabase.from_uri("postgresql://postgres:redhat@206.189.136.112:5432/Sample_Data_Text2SQL")
        return db

    if db_name == "db2":
        db = SQLDatabase.from_uri("postgresql://postgres:redhat@206.189.136.112:5432/Sample_Bank_Data")
        return db
    
    if db_name == "db3":
        db = SQLDatabase.from_uri("postgresql://postgres:redhat@206.189.136.112:5432/sample_crm_data")
        return db
    
    else:
        return None 

@app.route('/answer_sql', methods=['POST'])
def answer_sql():
    try:
        data = request.json
        user_question = data.get('question')
        db_name = data.get('database')

        if not data or not user_question or not db_name:
            return jsonify({"status":400, "error": "Invalid input: Question is missing"}), 200
        
        db = set_db_connection(db_name)

        if not db:
            return jsonify({"status":400, "error": "Couldn't connect to database"}), 200
            
        output = execute_question(user_question, db)

        if output.get('error'):
            return jsonify({"status":400, "error":"Unable to perform the desired query"}), 200
        else:
            output['status'] = 200
            return jsonify(output)
    
    except Exception as e:
        return jsonify({"status":200,"error": str(e)}), 500  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=84)