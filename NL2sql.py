import os
import socket
import logging
from datetime import datetime

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from flask import Flask, request, jsonify, redirect
import requests


OPENAI_KEY = os.getenv("OPENAI_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key = OPENAI_KEY)

logging.basicConfig(filename='NL2SQL_Database_activity.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

class DatabaseConnection:
    def __init__(self, username, password, hostname, port, database):
        try:
            self.db = SQLDatabase.from_uri(f"postgresql://{username}:{password}@{hostname}:{port}/{database}")
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            self.db = None

# Function to select the database
def select_db(username, password, hostname, port, database):
    return DatabaseConnection(username, password, hostname, port, database)
  
def get_db_config():
    try:
        api_endpoint = 'http://192.168.14.28:84/db_connection'  
        response = requests.get(api_endpoint)
        if response.status_code == 200:
            return response.json().get('db_config')
        else:
            print(f"Error: Failed to retrieve db_config from API endpoint. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error retrieving db_config: {e}")
        return None
  
def establish_db_connection():
    try:
        db_config = get_db_config()
        if db_config:
            return SQLDatabase.from_uri(
                f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['hostname']}:{db_config['port']}/{db_config['database']}"
            )
        else:
            print("Error: Failed to obtain db_config from the API endpoint.")
            return None
    except requests.RequestException as e:
        print(f"Request to API endpoint failed: {e}")
        return None
    except KeyError as e:
        print(f"Missing key in db_config JSON: {e}")
        return None
    except Exception as e:
        print(f"Error establishing database connection: {e}")
        return None
        
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
        return jsonify({'status':200, "error": "Please provide the correct credentials of database"}), 500

@app.route('/answer_sql', methods=['POST'])
def answer_sql():
    try:
        data = request.json
        user_question = data.get('question')
        # db_config = data.get('db_config')
        
        # db = select_db(**db_config).db
        db = establish_db_connection()

        if db is None:
            return jsonify({"status": 500, "error": "Failed to establish database connection."}), 500

        output = execute_question(user_question, db)
        
        if output.get('error'):
            return jsonify({"status": 400, "error": output['error']}), 200
        else:
            output['status'] = 200
            return jsonify(output)

    except Exception as e:
        return jsonify({"status": 500, "error": str(e)}), 500
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=84, debug=True)
    