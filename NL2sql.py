import os
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from flask import Flask, request, jsonify, redirect

OPENAI_KEY = os.getenv("OPENAI_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key = OPENAI_KEY)

def select_db(username, password, hostname, port, database):
        db = SQLDatabase.from_uri(f"postgresql://{username}:{password}@{hostname}:{port}/{database}")
        return db

    
app = Flask(__name__) 
def execute_question(question, db):
    try:
        output = dict()
        generate_query = create_sql_query_chain(llm, db)  
        execute_query = QuerySQLDataBaseTool(db=db)

        prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, provide the information of the user question.
            Question: {question}
            SQL Query: {query}
            SQL Table: {table}
            information: """
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

@app.route('/db_connection', methods = ['POST'])
def db_connections():
    try:
        data = request.json       
        hostname = data.get('hostname')
        if not hostname:
            return jsonify({'status':200, "error": "Please provide host name of the database"})
        
        port = data.get('port')
        if not port:
            return jsonify({'status': 200, "error": "Please provide port address of the database"})
        
        username = data.get('username')
        if not username:
            return jsonify({"status":200, "error": "Please provide the username of the database"})
        
        password = data.get('password')
        if not password:
            return jsonify({"status":200, "error":"Please provide the password of the database"})
        
        database = data.get('database')
        
        if not database:
            return jsonify({"status":200, "error":"Please provide the name of the database"})
        
        db = select_db(username, password, hostname, port, database)

        if not db:
            return jsonify({"status": 200, "message": "Connection could not established successfully to PostgreSQL database!"})
        else:
           return ({"status":200, "message": "Connection established successfully"})
           
    except Exception as e:
        return jsonify({'status':200, "error": "Please provide the correct credentials of database"}), 400

@app.route('/answer_sql', methods=['POST'])
def answer_sql():
    try:
        data = request.json
        user_question = data.get('question')
    
        output = execute_question(user_question, select_db.db)
        
        if output.get('error'):
            return jsonify({"status":400, "error":"Unable to perform the desired query"}), 200
        else:
            output['status'] = 200
            return jsonify(output)
    
    except Exception as e:
        return jsonify({"status":200,"error": str(e)}), 500  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=84, debug=True)
    