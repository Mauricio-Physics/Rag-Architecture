from langchain_community.document_loaders import TextLoader
import psycopg2
from langchain_text_splitters import TokenTextSplitter
import os, glob
import re


class DataIngestion:
    def __init__(self, retriever, documents_path):
        self.retriever = retriever
        self.conn = psycopg2.connect(database=os.environ['PSQL_DB'],
                            user=os.environ['PSQL_USERNAME'],
                            host=os.environ['PSQL_HOST'],
                            password=os.environ['PSQL_PASSWORD'],
                            port=os.environ['PSQL_PORT'])
        self.chunk_size=os.environ['CHUNK_SIZE']
        self.overlap=os.environ['OVERLAP']
        self.documents_path = documents_path

    def preprocess_data(self):
        cursor = self.conn.cursor()
        files = []
        for file in glob.glob(self.documents_path + "*.md"):
            files.append(file.replace(self.documents_path, ''))
        print(files)
        query_machine_code = 'SELECT d.document_machine FROM documents d where document_filename like \'{doc_name}%\''
        tags_list = []

        for i in files:
            sql_query = query_machine_code.format(doc_name=i[:-3])
            cursor.execute(sql_query)
            results = cursor.fetchall()[0][0]
            tags_list += [results]
        docs = []
        tags = []
        for i in range(len(files)):
            loader = TextLoader(self.documents_path + files[i])
            documents = loader.load()
            text_splitter = TokenTextSplitter(chunk_size=int(self.chunk_size), chunk_overlap=int(self.overlap))
            docs_temp = text_splitter.split_documents(documents)
            for j in docs_temp:
                j.metadata['tag'] = tags_list[i]
                j.metadata['file'] = files[i]
                j.page_content = j.page_content.replace('<figure>\n', '')
                j.page_content = j.page_content.replace('</figure>\n', '')
                j.page_content = re.sub('!\[]\(figures/.*?\)', '', j.page_content, flags=re.DOTALL)
            docs += docs_temp
        return docs

    def run(self):
        docs = self.preprocess_data()
        self.retriever.add_texts(docs)