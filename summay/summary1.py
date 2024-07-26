import asyncio
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnable
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Any, Optional, Dict
from flask import jsonify
import ollama
import os

from langchain_core.prompt_values import StringPromptValue

# client = Client(host='http://35.244.5.146:11434')

class OllamaWrapper(Runnable):
    def __init__(self, model_name):
        self.model_name = model_name

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        if isinstance(input, dict) and 'prompt' in input:
            prompt = input['prompt']
        elif isinstance(input, StringPromptValue):
            prompt = input.to_string()
        elif isinstance(input, str):
            prompt = input
        else:
            prompt = str(input)
        
        response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        return self.invoke(input, config)

class DocumentProcessor:
    def __init__(self, batch_size=5):
        self.llm = OllamaWrapper('mistral')
        self.restoration_template = PromptTemplate.from_template(self._get_restoration_template())
        self.summary_template = PromptTemplate.from_template(self._get_summary_template())
        
        self.restoration_chain = self.restoration_template | self.llm | StrOutputParser()
        self.summary_chain = self.summary_template | self.llm | StrOutputParser()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.batch_size = batch_size

    @staticmethod
    def _get_restoration_template():
        return """<s>[INST] The following is an article:
        {text}
        Based on this, please restore the content of the document by removing anomalous characters and gibberish to produce comprehensible content with proper grammar. 
        Answer:  [/INST] </s>"""

    @staticmethod
    def _get_summary_template():
        return """<s>[INST] The following is an article:
        {text}
        Based on this, summarize the document with relevant facts. Don't generate irrelevant summary. 
        Answer:  [/INST] </s>"""

    def load_documents(self, file_path):
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")
        return loader.load()

    def _get_token_count(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def _split_text(self, text):
        return self.text_splitter.split_text(text)

    async def _process_chunk(self, chunk: str, process_type: str) -> str:
        if process_type == "restore":
            result = await self.restoration_chain.ainvoke({"text": chunk})
        elif process_type == "summarize":
            result = await self.summary_chain.ainvoke({"text": chunk})
        return result

    async def _process_batch(self, batch: List[str], process_type: str) -> List[str]:
        tasks = [self._process_chunk(chunk, process_type) for chunk in batch]
        return await asyncio.gather(*tasks)

    async def _combine_summaries(self, summaries):
        combined_summary = " ".join(summaries)
        final_summary = await self.summary_chain.ainvoke({"text": combined_summary})
        return final_summary

    async def process_and_summarize_file(self, folder_path: str, restoration: str):
        file_summaries = []
        folder_full_text = ""

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if file_path.lower().endswith(('.pdf', '.txt')):
                docs = self.load_documents(file_path)
                file_text = " ".join([doc.page_content for doc in docs])
                token_count = self._get_token_count(file_text)
                folder_full_text += file_text + " "

                summary = ""  # Initialize summary variable
                if token_count <= 8000:
                    if restoration == 'True':
                        restored_content = await self.restoration_chain.ainvoke({"text": file_text})
                        summary = await self.summary_chain.ainvoke({"text": restored_content})
                    elif restoration == "False":
                        summary = await self.summary_chain.ainvoke({"text": file_text})
                    else:
                        return jsonify({"error": "Restoration must be either True or False", "status": 500}),200
                else:
                    chunks = self._split_text(file_text)
                    batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
                    
                    if restoration == "True":
                        restore_tasks = [self._process_batch(batch, "restore") for batch in batches]
                        restored_chunks = await asyncio.gather(*restore_tasks)
                        restored_chunks = [chunk for batch in restored_chunks for chunk in batch]

                        summary_batches = [restored_chunks[i:i + self.batch_size] for i in range(0, len(restored_chunks), self.batch_size)]
                        summarize_tasks = [self._process_batch(batch, "summarize") for batch in summary_batches]
                        summaries = await asyncio.gather(*summarize_tasks)
                        summaries = [summary for batch in summaries for summary in batch]
                    elif restoration == "False":
                        summarize_tasks = [self._process_batch(batch, "summarize") for batch in batches]
                        summaries = await asyncio.gather(*summarize_tasks)
                        summaries = [summary for batch in summaries for summary in batch]
                    else:
                        return jsonify({"error": "Restoration must be either True or False", "status": 500}),200
                    
                    summary = await self._combine_summaries(summaries)

                file_summaries.append({
                    "filename": filename,
                    "summary": summary,
                    "token_count": token_count
                })

        # Process folder summary
        folder_token_count = self._get_token_count(folder_full_text)
        folder_summary = ""  # Initialize folder_summary variable
        if folder_token_count <= 8000:
            if restoration == 'True':
                restored_content = await self.restoration_chain.ainvoke({"text": folder_full_text})
                folder_summary = await self.summary_chain.ainvoke({"text": restored_content})
            elif restoration == "False":
                folder_summary = await self.summary_chain.ainvoke({"text": folder_full_text})
            else:
                return jsonify({"error": "Restoration must be either True or False", "status": 500}),200
        else:
            chunks = self._split_text(folder_full_text)
            batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
            
            if restoration == "True":
                restore_tasks = [self._process_batch(batch, "restore") for batch in batches]
                restored_chunks = await asyncio.gather(*restore_tasks)
                restored_chunks = [chunk for batch in restored_chunks for chunk in batch]

                summary_batches = [restored_chunks[i:i + self.batch_size] for i in range(0, len(restored_chunks), self.batch_size)]
                summarize_tasks = [self._process_batch(batch, "summarize") for batch in summary_batches]
                summaries = await asyncio.gather(*summarize_tasks)
                summaries = [summary for batch in summaries for summary in batch]
            elif restoration == "False":
                summarize_tasks = [self._process_batch(batch, "summarize") for batch in batches]
                summaries = await asyncio.gather(*summarize_tasks)
                summaries = [summary for batch in summaries for summary in batch]
            else:
                return jsonify({"error": "Restoration must be either True or False", "status": 500}),200
            
            folder_summary = await self._combine_summaries(summaries)

        return {
            "file_summaries": file_summaries,
            "folder_summary": folder_summary,
            "folder_token_count": folder_token_count
        }
