# # # app.py
# # from flask import Flask, request, jsonify
# # import torch
# # import sqlparse
# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # app = Flask(__name__)

# # # Initialize the model and tokenizer
# # model_name = "defog/llama-3-sqlcoder-8b"
# # cache_dir = 'E:\\AI_ML\\text2sql\\Huggingface_models\\models--defog--llama-3-sqlcoder-8b'
# # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# # # Load model in 4-bit quantized form for GPU
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     trust_remote_code=True,
# #     load_in_4bit=True,
# #     device_map="auto" if device == "cuda" else None,
# #     use_cache=True,
# #     cache_dir=cache_dir,
# # )

# # # Define DDL statements as a global variable
# # ddl_statements = ""
# # print(ddl_statements)
# # # Define the prompt template
# # prompt_template = """User

# # Generate a SQL query to answer this question: `{question}`

# # DDL statements:

# # {ddl_statements}

# # The following SQL query best answers the question `{question}`:
# # ```sql
# # """
# # def generate_query(question):
# #     if not ddl_statements:
# #         return "DDL statements not provided. Please upload a DDL file first."

# #     updated_prompt = prompt_template.format(question=question, ddl_statements=ddl_statements)
# #     inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
# #     generated_ids = model.generate(
# #         **inputs,
# #         num_return_sequences=1,
# #         eos_token_id=tokenizer.eos_token_id,
# #         pad_token_id=tokenizer.eos_token_id,
# #         max_new_tokens=400,
# #         do_sample=False,
# #         num_beams=1,
# #         temperature=0.0,
# #         top_p=1,
# #     )
# #     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# #     torch.cuda.empty_cache()  # Clear GPU memory
# #     torch.cuda.synchronize()  # Synchronize GPU

# #     # Extract SQL query from the output more robustly
# #     try:
# #         sql_query = outputs[0].split("```sql")[1].split("```")[0].strip()
# #     except IndexError:
# #         sql_query = "Could not extract SQL query. Ensure the prompt and model are configured correctly."

# #     return sql_query

# # # def generate_query(question):
# #     if not ddl_statements:
# #         return "DDL statements not provided. Please upload a DDL file first."

# #     updated_prompt = prompt_template.format(question=question, ddl_statements=ddl_statements)
# #     inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
# #     generated_ids = model.generate(
# #         **inputs,

# #         num_return_sequences=1,
# #         eos_token_id=tokenizer.eos_token_id,
# #         pad_token_id=tokenizer.eos_token_id,
# #         max_new_tokens=400,
# #         do_sample=False,
# #         num_beams=1,
# #         temperature=0.0,
# #         top_p=1,
# #     )
# #     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# #     torch.cuda.empty_cache()  # Clear GPU memory
# #     torch.cuda.synchronize()  # Synchronize GPU

# #     # Extract SQL query from the output
# #     sql_query = outputs[0].split("```sql")[1].split("```")[0].strip()
# #     return sql_query

# # @app.route('/query', methods=['POST'])
# # def query_endpoint():
# #     data = request.get_json()
# #     question = data.get('question')
# #     if not question:
# #         return jsonify({"error": "Question not provided"}), 400

# #     generated_sql = generate_query(question)

# #     return jsonify({
# #         "question": question,
# #         "sql": sqlparse.format(generated_sql, reindent=True)
# #     })

# # @app.route('/upload-ddl', methods=['POST'])
# # def upload_ddl():
# #     global ddl_statements

# #     if 'file' not in request.files:
# #         return jsonify({"error": "No file part"}), 400

# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({"error": "No selected file"}), 400

# #     if file and file.filename.endswith('.ddl'):
# #         ddl_statements = file.read().decode('utf-8')
# #         # print(ddl_statements)
# #         return jsonify({"message": "DDL statements updated successfully"}), 200
# #     else:
# #         return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400

# # if __name__ == '__main__':
# #     # Uncomment and configure ngrok if needed
# #     # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
# #     # ngrok_tunnel = ngrok.connect(5000)
# #     # print("Public URL:", ngrok_tunnel.public_url)
# #     app.run(host='0.0.0.0', port=5000, debug=True)

# # app.py
# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import os

# app = Flask(__name__)

# # Initialize the model and tokenizer
# model_name = "defog/llama-3-sqlcoder-8b"
# cache_dir = 'E:\\AI_ML\\text2sql\\Huggingface_models\\models--defog--llama-3-sqlcoder-8b'

# # Ensure cache directory exists
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)

# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# # Load model in 4-bit quantized form for GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     load_in_4bit=True,
#     device_map="auto" if device == "cuda" else None,
#     use_cache=True,
#     cache_dir=cache_dir,
# )

# # Define DDL statements as a global variable
# ddl_statements = ""

# # Define the prompt template
# prompt_template = """User

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     if not ddl_statements:
#         return "DDL statements not provided. Please upload a DDL file first."

#     updated_prompt = prompt_template.format(question=question, ddl_statements=ddl_statements)
#     inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
#     generated_ids = model.generate(
#         **inputs,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=400,
#         do_sample=False,
#         num_beams=1,
#         temperature=0.0,
#         top_p=1,
#     )
#     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     torch.cuda.empty_cache()  # Clear GPU memory

#     # Extract SQL query from the output more robustly
#     try:
#         sql_query = outputs[0].split("```sql")[1].split("```")[0].strip()
#     except IndexError:
#         sql_query = "Could not extract SQL query. Ensure the prompt and model are configured correctly."

#     return sql_query

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     data = request.get_json()
#     question = data.get('question')
#     if not question:
#         return jsonify({"error": "Question not provided"}), 400

#     generated_sql = generate_query(question)

#     return jsonify({
#         "question": question,
#         "sql": sqlparse.format(generated_sql, reindent=True)
#     })

# @app.route('/upload-ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and file.filename.endswith('.ddl'):
#         ddl_statements = file.read().decode('utf-8')
#         return jsonify({"message": "DDL statements updated successfully"}), 200
#     else:
#         return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400

# if __name__ == '__main__':
#     # Uncomment and configure ngrok if needed
#     # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
#     # ngrok_tunnel = ngrok.connect(5000)
#     # print("Public URL:", ngrok_tunnel.public_url)
# app.py
# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:", device)
# # Set quantized engine to FBGEMM
# torch.backends.quantized.engine = 'fbgemm'
# # Your quantization and inference code here
# # model_name = "defog/llama-3-sqlcoder-8b"
# model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the model in 4-bit quantized form
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     # cache_dir= r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\blobs', #cache_dir
#     trust_remote_code=True,
#     load_in_4bit=True,
#     device_map="auto",
#     use_cache=True,
#     # self.model_name,
#     #             trust_remote_code=True,
#     #             torch_dtype=torch.float16,
#     #             device_map="auto",
#     #             use_cache=True,
#     #             # cache_dir=self.cache_dir,
# )

# app = Flask(__name__)

# # Global variable to store DDL
# ddl_statements = ""

# prompt = """user

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
#     inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
#     generated_ids = model.generate(
#         **inputs,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=400,
#         do_sample=False,
#         num_beams=1,
#         temperature=0.0,
#         top_p=1,
#     )
#     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     return outputs[0].split("```sql")[1].split(";")[0]

# @app.route('/upload_ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and file.filename.endswith('.ddl'):
#         ddl_statements = file.read().decode('utf-8')
#         return jsonify({"message": "DDL statements updated successfully"}), 200
#     else:
#         return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     data = request.get_json()
#     question = data.get('question')
#     if not question:
#         return jsonify({"error": "Question not provided"}), 400

#     generated_sql = generate_query(question)
#     return jsonify({
#         "query": question,
#         "sql": sqlparse.format(generated_sql, reindent=True)
#     })

# if __name__ == '__main__':
#     # app.run(host='0.0.0.0', port=5000)

#     app.run(host='0.0.0.0', port=5000, debug=True)

# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:", device)

# # Set quantized engine to FBGEMM
# torch.backends.quantized.engine = 'fbgemm'

# # Model path
# model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load the model in 4-bit quantized form
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     load_in_4bit=True,
#     device_map="auto",
#     use_cache=True,
# )

# app = Flask(__name__)

# # Global variable to store DDL
# ddl_statements = ""

# prompt = """user

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
#     inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
#     generated_ids = model.generate(
#         **inputs,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=400,
#         do_sample=False,
#         num_beams=1,
#         temperature=0.0,
#         top_p=1,
#     )
#     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()
#     return outputs[0].split("```sql")[1].split(";")[0].strip()

# @app.route('/upload_ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements

#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and file.filename.endswith('.ddl'):
#         ddl_statements = file.read().decode('utf-8')
#         return jsonify({"message": "DDL statements updated successfully"}), 200
#     else:
#         return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     global ddl_statements
    
#     if not ddl_statements:
#         return jsonify({"error": "DDL statements not provided"}), 400

#     data = request.get_json()
#     question = data.get('question')
#     if not question:
#         return jsonify({"error": "Question not provided"}), 400

#     generated_sql = generate_query(question)
#     formatted_sql = sqlparse.format(generated_sql, reindent=True).replace('\n', ' ').strip()
#     return jsonify({
#         "query": question,
#         "sql": formatted_sql
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

# Added exception handling # 
# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device:", device)

# # Set quantized engine to FBGEMM
# torch.backends.quantized.engine = 'fbgemm'

# # Model path
# model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'

# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
# except Exception as e:
#     logging.error("Failed to load tokenizer: %s", e)
#     raise

# try:
#     # Load the model in 4-bit quantized form
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         load_in_4bit=True,
#         device_map="auto",
#         use_cache=True,
#     )
# except Exception as e:
#     logging.error("Failed to load model: %s", e)
#     raise

# app = Flask(__name__)

# # Global variable to store DDL
# ddl_statements = ""

# prompt = """user

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     try:
#         updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
#         inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
#         generated_ids = model.generate(
#             **inputs,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             max_new_tokens=400,
#             do_sample=False,
#             num_beams=1,
#             temperature=0.0,
#             top_p=1,
#         )
#         outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         return outputs[0].split("```sql")[1].split(";")[0].strip()
#     except Exception as e:
#         logging.error("Failed to generate query: %s", e)
#         raise

# @app.route('/upload_ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements

#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         if file and file.filename.endswith('.ddl'):
#             ddl_statements = file.read().decode('utf-8')
#             return jsonify({"message": "DDL statements updated successfully"}), 200
#         else:
#             return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400
#     except Exception as e:
#         logging.error("Failed to upload DDL: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     global ddl_statements

#     try:
#         if not ddl_statements:
#             return jsonify({"error": "DDL statements not provided"}), 400

#         data = request.get_json()
#         question = data.get('question')
#         if not question:
#             return jsonify({"error": "Question not provided"}), 400

#         generated_sql = generate_query(question)
#         formatted_sql = sqlparse.format(generated_sql, reindent=True).replace('\n', ' ').strip()
#         return jsonify({
#             "query": question,
#             "sql": formatted_sql
#         })
#     except Exception as e:
#         logging.error("Failed to process query: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True)
#     except Exception as e:
#         logging.error("Failed to start server: %s", e)
#         raise


# Optimized code query performance not working

# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info("device: %s", device)

# # Set quantized engine to FBGEMM
# torch.backends.quantized.engine = 'fbgemm'

# # Model path
# model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'

# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     logging.info("Tokenizer loaded successfully")
# except Exception as e:
#     logging.error("Failed to load tokenizer: %s", e)
#     raise

# try:
#     # Load the model in 4-bit quantized form
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         load_in_4bit=True,
#         device_map="auto",
#         use_cache=True,
#     )
#     model.to(device)
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error("Failed to load model: %s", e)
#     raise

# app = Flask(__name__)

# # Global variable to store DDL
# ddl_statements = ""

# prompt = """user

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     try:
#         updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
#         inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
#         generated_ids = model.generate(
#             **inputs,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             max_new_tokens=400,
#             do_sample=False,
#             num_beams=1,
#             temperature=0.0,
#             top_p=1,
#         )
#         outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         return outputs[0].split("```sql")[1].split(";")[0].strip()
#     except Exception as e:
#         logging.error("Failed to generate query: %s", e)
#         raise
#     finally:
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

# @app.route('/upload_ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         if file and file.filename.endswith('.ddl'):
#             ddl_statements = file.read().decode('utf-8')
#             return jsonify({"message": "DDL statements updated successfully"}), 200
#         else:
#             return jsonify({"error": "Invalid file type. Please upload a .ddl file"}), 400
#     except Exception as e:
#         logging.error("Failed to upload DDL: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     global ddl_statements
#     try:
#         if not ddl_statements:
#             return jsonify({"error": "DDL statements not provided"}), 400

#         data = request.get_json()
#         question = data.get('question')
#         if not question:
#             return jsonify({"error": "Question not provided"}), 400

#         generated_sql = generate_query(question)
#         formatted_sql = sqlparse.format(generated_sql, reindent=True).replace('\n', ' ').strip()
#         return jsonify({
#             "query": question,
#             "sql": formatted_sql
#         })
#     except Exception as e:
#         logging.error("Failed to process query: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True)
#     except Exception as e:
#         logging.error("Failed to start server: %s", e)
#         raise


# Handle large DDLs

# from flask import Flask, request, jsonify
# import torch
# import sqlparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import logging
# from werkzeug.utils import secure_filename
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logging.info("device: %s", device)

# # Set quantized engine to FBGEMM
# torch.backends.quantized.engine = 'fbgemm'

# # Model path
# model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'

# try:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     logging.info("Tokenizer loaded successfully")
# except Exception as e:
#     logging.error("Failed to load tokenizer: %s", e)
#     raise

# try:
#     # Load the model in 4-bit quantized form
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         trust_remote_code=True,
#         load_in_4bit=True,
#         device_map="auto",
#         use_cache=True,
#     )
#     model.to(device)
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error("Failed to load model: %s", e)
#     raise

# app = Flask(__name__)

# # Global variable to store DDL
# ddl_statements = ""
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# prompt = """user

# Generate a SQL query to answer this question: `{question}`

# DDL statements:

# {ddl_statements}

# The following SQL query best answers the question `{question}`:
# ```sql
# """

# def generate_query(question):
#     try:
#         updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
#         inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
#         generated_ids = model.generate(
#             **inputs,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             max_new_tokens=400,
#             do_sample=False,
#             num_beams=1,
#             temperature=0.0,
#             top_p=1,
#         )
#         outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         return outputs[0].split("```sql")[1].split(";")[0].strip()
#     except Exception as e:
#         logging.error("Failed to generate query: %s", e)
#         raise
#     finally:
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

# @app.route('/upload_ddl', methods=['POST'])
# def upload_ddl():
#     global ddl_statements
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         with open(file_path, 'r', encoding='utf-8') as f:
#             ddl_statements = f.read()

#         os.remove(file_path)
#         return jsonify({"message": "DDL statements updated successfully"}), 200
#     except Exception as e:
#         logging.error("Failed to upload DDL: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     global ddl_statements
#     try:
#         if not ddl_statements:
#             return jsonify({"error": "DDL statements not provided"}), 400

#         data = request.get_json()
#         question = data.get('question')
#         if not question:
#             return jsonify({"error": "Question not provided"}), 400

#         generated_sql = generate_query(question)
#         formatted_sql = sqlparse.format(generated_sql, reindent=True).replace('\n', ' ').strip()
#         return jsonify({
#             "query": question,
#             "sql": formatted_sql
#         })
#     except Exception as e:
#         logging.error("Failed to process query: %s", e)
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True)
#     except Exception as e:
#         logging.error("Failed to start server: %s", e)
#         raise


# Scale for multiple users

from flask import Flask, request, jsonify
import torch
import sqlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("device: %s", device)

# Set quantized engine to FBGEMM
torch.backends.quantized.engine = 'fbgemm'

# Model path
model_name = r'E:\AI_ML\text2sql\Huggingface_models\models--defog--llama-3-sqlcoder-8b\snapshots\0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Tokenizer loaded successfully")
except Exception as e:
    logging.error("Failed to load tokenizer: %s", e)
    raise

try:
    # Load the model in 4-bit quantized form
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto",
        use_cache=True,
    )
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Failed to load model: %s", e)
    raise

app = Flask(__name__)

# Global variable to store DDL
ddl_statements = ""
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

prompt = """user

Generate a SQL query to answer this question: `{question}`

DDL statements:

{ddl_statements}

The following SQL query best answers the question `{question}`:
```sql
"""

def generate_query(question):
    try:
        updated_prompt = prompt.format(question=question, ddl_statements=ddl_statements)
        inputs = tokenizer(updated_prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            top_p=1,
        )
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0].split("```sql")[1].split(";")[0].strip()
    except Exception as e:
        logging.error("Failed to generate query: %s", e)
        raise
    finally:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

@app.route('/upload_ddl', methods=['POST'])
def upload_ddl():
    global ddl_statements
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with open(file_path, 'r') as f:
            ddl_statements = f.read()

        os.remove(file_path)
        return jsonify({"message": "DDL statements updated successfully"}), 200
    except Exception as e:
        logging.error("Failed to upload DDL: %s", e)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/query', methods=['POST'])
def query_endpoint():
    global ddl_statements
    try:
        if not ddl_statements:
            return jsonify({"error": "DDL statements not provided"}), 400

        data = request.get_json()
        question = data.get('question')
        if not question:
            return jsonify({"error": "Question not provided"}), 400

        generated_sql = generate_query(question)
        formatted_sql = sqlparse.format(generated_sql, reindent=True).replace('\n', ' ').strip()
        return jsonify({
            "query": question,
            "sql": formatted_sql
        })
    except Exception as e:
        logging.error("Failed to process query: %s", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error("Failed to start server: %s", e)
        raise


# Improve query accuracy