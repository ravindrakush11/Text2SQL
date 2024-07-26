from flask import Flask, request, jsonify, send_file, abort, Response
from summary1 import DocumentProcessor
from werkzeug.utils import secure_filename
import asyncio
from werkzeug.exceptions import BadRequest
import traceback
from flask_cors import CORS
import logging
import os
import shutil
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

document_processor = DocumentProcessor(batch_size=5)
upload_folder = "/mnt/summary/upload"
output_folder = "/mnt/summary/output" 

@app.route('/summarize_file', methods=['POST'])
def summarize_file():
    logger.info("Starting summarize_file function")
    try:
        folder_name = request.form['folder_name']
        files = request.files.getlist('files[]')
        restoration = request.form.get('restoration')
        
        logger.info(f"Received request: folder_name={folder_name}, restoration={restoration}, number of files={len(files)}")
    
        if 'folder_name' not in request.form or not files or not restoration:
            logger.warning("Missing required parameters")
            return jsonify({"error": "Folder name/files/restoration not provided", "status": 400}), 200
    
        upload_folder_path = os.path.join(upload_folder, folder_name)
        if not os.path.exists(upload_folder_path):
            logger.info(f"Creating upload folder: {upload_folder_path}")
            os.makedirs(upload_folder_path)
    
        # Save uploaded files
        logger.info("Saving uploaded files")
        for file in files:
            if file and file.filename:
                if file.filename.lower().endswith(('.pdf', '.txt')):
                    file_path = os.path.join(upload_folder_path, file.filename)
                    file.save(file_path)
                    logger.info(f"Saved file: {file_path}")
    
        if not os.listdir(upload_folder_path):
            logger.warning("No valid files uploaded")
            return jsonify({"error": "No valid files uploaded", "status": 400}), 200
    
        # Run the asynchronous function in a synchronous context
        logger.info("Processing and summarizing files")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(document_processor.process_and_summarize_file(upload_folder_path, restoration))
        finally:
            loop.close()
    
        # Save the summary to the download folder
        logger.info("Saving summary to download folder")
        download_folder_path = os.path.join(output_folder, folder_name)
        if not os.path.exists(download_folder_path):
            os.makedirs(download_folder_path)
        
        summary_file_path = os.path.join(download_folder_path, f"{folder_name}_summary.json")
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Summary saved to: {summary_file_path}")
    
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

@app.route('/download_folder', methods=['POST'])
def download_folder():
    logger.info("Starting download_folder function")
    try:
        data = request.json
        folder_name = data.get('folder_name')
        logger.info(f"Received request for folder: {folder_name}")
        
        if not folder_name:
            logger.warning("Folder name not provided")
            return jsonify({"error": "Folder name not provided", "status" : 400}), 200
        
        folder_path = os.path.join(output_folder, folder_name)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            logger.warning(f"Invalid folder path: {folder_path}")
            return jsonify({"error": "Invalid folder path", "status" : 400}), 200
        
        files = os.listdir(folder_path)
        if not files:
            logger.warning(f"Folder is empty: {folder_path}")
            return jsonify({"error": "Folder is empty", "status" : 404}), 200
        
        file_path = os.path.join(folder_path, files[0])
        if not os.path.isfile(file_path):
            logger.warning(f"No valid file found in the folder: {folder_path}")
            return jsonify({"error": "No valid file found in the folder", "status" : 404}), 200
        
        logger.info(f"Sending file: {file_path}")
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

@app.route("/clear_folder/", methods=["POST"])
def clear_folder():
    logger.info("Starting clear_folder function")
    try:
        data = request.get_json()
        logger.info(f"Received request: {data}")

        if not data or "folder_type" not in data:
            logger.warning("Folder type not provided")
            return jsonify({"error": "Folder type not provided", "status": 400}), 200

        folder_type = data["folder_type"]
        folder_name = data.get("folder_name")

        messages = []

        if folder_type not in ["upload", "output", "both"]:
            logger.warning(f"Invalid folder type provided: {folder_type}")
            return jsonify({"error": "Invalid folder type provided", "status": 400}), 200

        if folder_type == "upload" or folder_type == "both":
            if folder_name:
                folder_path = os.path.join(upload_folder, folder_name)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    messages.append(f"Deleted specific upload folder: {folder_path}")
                    logger.info(f"Deleted specific upload folder: {folder_path}")
                else:
                    logger.warning(f"Specified upload folder does not exist: {folder_path}")
                    return jsonify({"error": "Specified upload folder does not exist", "status": 400}), 200
            else:
                if os.path.exists(upload_folder):
                    shutil.rmtree(upload_folder)
                    os.makedirs(upload_folder)
                    messages.append(f"Cleared entire upload folder: {upload_folder}")
                    logger.info(f"Cleared entire upload folder: {upload_folder}")

        if folder_type == "output" or folder_type == "both":
            if folder_name:
                folder_path = os.path.join(output_folder, folder_name)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                    messages.append(f"Deleted specific output folder: {folder_path}")
                    logger.info(f"Deleted specific output folder: {folder_path}")
                else:
                    logger.warning(f"Specified output folder does not exist: {folder_path}")
                    return jsonify({"error": "Specified output folder does not exist", "status": 400}), 200
            else:
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                    os.makedirs(output_folder)
                    messages.append(f"Cleared entire output folder: {output_folder}")
                    logger.info(f"Cleared entire output folder: {output_folder}")

        logger.info(f"Clear folder operation completed: {' '.join(messages)}")
        return jsonify({"message": " ".join(messages), "status": 200}), 200
    except Exception as e:
        logger.error(f"Error in clear_folder: {e}")
        return jsonify({"error": str(e), "status": 500}), 200

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5005)