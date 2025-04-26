# Text-to-SQL: Natural Language Querying for Databases

## ✨ About This Project

Welcome to the **Text-to-SQL** project! 🚀 This repository enables you to effortlessly convert natural language queries into structured **SQL** commands, allowing users to interact with databases without needing to write complex SQL syntax.

Leveraging **LangChain** and **OpenAI**, this project translates user questions into meaningful SQL queries that can be executed on PostgreSQL databases. This tool makes database querying as simple as asking a question in plain English.

### 🌟 Key Features:
- **Natural Language to SQL**: Convert questions like "Show me the top 10 products by price" into SQL queries.
- **Integration with PostgreSQL**: Connects to PostgreSQL databases, runs SQL queries, and retrieves results.
- **Flask API**: Exposes a simple REST API for submitting questions and receiving answers in real-time.
- **Dynamic Database Connections**: Establish database connections on-the-fly using credentials passed via API.

---

## 🧠 How It Works

This project uses **LangChain** and **OpenAI's GPT-3.5** model to:
1. **Parse user questions** and generate corresponding SQL queries.
2. **Execute these queries** on a connected PostgreSQL database.
3. **Return the results** to the user, rephrased for clarity and relevance.

It supports dynamic interaction with any PostgreSQL database, making it ideal for building intelligent applications, chatbots, and data assistants.

### 🛠️ Tech Stack:
- **LangChain** for query generation
- **OpenAI GPT-3.5** for natural language processing
- **Flask** for serving a REST API
- **PostgreSQL** for database interactions
- **Requests** for API-based configuration retrieval

---

## 🧪 How to Use

### 1. **Clone the Repository**

```bash
git clone https://github.com/ravindrakush11/Text2SQL.git
cd Text2SQL
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**

You need to set up the following environment variables:
- `OPENAI_KEY` — Your OpenAI API key
- `LANGCHAIN_TRACING_V2` — Optional: Set this for LangChain tracing
- `LANGCHAIN_API_KEY` — Optional: For additional LangChain configuration

### 4. **Start the Flask Application**

Run the following command to start the Flask server:

```bash
python app.py
```

By default, the server runs on **port 84**. You can now interact with the application via the API.

---

## 📝 API Endpoints

### 1. **/db_connection** [POST]

Use this endpoint to establish a connection to your PostgreSQL database by providing the necessary credentials.

**Request Body:**
```json
{
    "hostname": "your-db-hostname",
    "port": "5432",
    "username": "your-db-username",
    "password": "your-db-password",
    "database": "your-db-name"
}
```

**Response:**
```json
{
    "status": 200,
    "message": "Connection established successfully"
}
```

### 2. **/answer_sql** [POST]

Submit a user question to be transformed into an SQL query and executed on the connected PostgreSQL database.

**Request Body:**
```json
{
    "question": "What are the top 5 products by sales?"
}
```

**Response:**
```json
{
    "status": 200,
    "info": {
        "query": "SELECT * FROM products ORDER BY sales DESC LIMIT 5",
        "table": "products"
    },
    "information": "Here are the top 5 products by sales..."
}
```

---

## ⚙️ Configuration

The database connection is established by retrieving credentials from an API endpoint or by directly passing the connection details. 

1. Modify the **`get_db_config()`** function to fetch database credentials from a different source or API if needed.
2. You can also tweak the **prompt templates** in the `execute_question()` function to adjust how the SQL query is formatted or refined.


