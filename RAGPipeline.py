from dotenv import load_dotenv
from lib.utilities import setup_chat_model, create_load_retriever
import os
from langchain_core.prompts import ChatPromptTemplate
import re
import psycopg2


class RAGPiepeline():
    def __init__(self, system_prompt):
        load_dotenv()
        self.system_prompt = system_prompt
        self.model_name = os.environ['MODEL_NAME']
        self.llm = setup_chat_model(model_name=self.model_name)
        self.retriever = None
        self.index_name = os.environ['ELASTICSEARCH_INDEX_NAME']
        self.initialize_retriever()
        self.conn = psycopg2.connect(database=os.environ['PSQL_DB'],
                            user=os.environ['PSQL_USERNAME'],
                            host=os.environ['PSQL_HOST'],
                            password=os.environ['PSQL_PASSWORD'],
                            port=os.environ['PSQL_PORT'])

    def initialize_retriever(self):
        if self.index_name is not None:
            elasticsearch_url = 'http://{elastic_username}:{elastic_password}@{elastic_host}:{elastic_port}'.format(
                elastic_username=os.environ['ELASTICSEARCH_USERNAME'],
                elastic_password=os.environ['ELASTICSEARCH_PASSWORD'],
                elastic_host=os.environ['ELASTICSEARCH_HOST'],
                elastic_port=os.environ['ELASTICSEARCH_PORT']
            )
            recreate = False
            self.retriever = create_load_retriever(elasticsearch_url, self.index_name.lower(), recreate=recreate,
                                                   embedding_model_name=None)

    def retrive_documents(self, query, machine_code, n_retrieved=3):
        retrieved_chunks = self.retriever.retrieve(query=query, machine_code=machine_code, n_retrieved=n_retrieved)
        return retrieved_chunks

    def extract_machine_name(self, query):
        system = (f"Data la domanda in input estrai il nome della macchina. La risposta fornita deve essere del tipo"
                  f"'[Nome macchine]'"
                  f"Dove 'Nome macchine' è il nome che ti viene richiesto di estrarre. Non aggiungere testo aggiuntivo,"
                  f"riporta solo il nome della macchina come precedentemente descritto. Il nome deve essere riportato"
                  f"fedelmente e in lingua italiana. Riporta il nome tra parentesi quadre")
        user = query
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.llm
        answer = chain.invoke({}).content.replace('(', '[').replace(')', ']')
        return re.findall("\[(.*?)\]", answer)[0].replace("'","")

    def extract_machine_code(self, machine_name):
        query_machine_code = f'SELECT m.machine_code FROM machines m where machine_description like \'{machine_name}%\''
        cursor = self.conn.cursor()
        cursor.execute(query_machine_code)
        results = cursor.fetchall()[-1][0]
        return results

    def create_rag_prompt(self, query, machine_code, n_retrieved=3):
        docs = self.retrive_documents(query=query, machine_code=machine_code, n_retrieved=n_retrieved)
        to_return_string = ''
        for i in docs:
            to_return_string += i.page_content + '\n\n'
        return to_return_string

    def generate_answer(self, rag_prompt, question):
        system = f"###Istruzione\n{self.system_prompt}\n\n###Contesto\n{rag_prompt}"
        user = '###Domanda\n'+question
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
        chain = prompt | self.llm
        return chain.invoke({})

    def run(self, user_question):
        machine_name = self.extract_machine_name(query=user_question)
        machine_code = self.extract_machine_code(machine_name=machine_name)
        rag_prompt = self.create_rag_prompt(query=user_question, machine_code=machine_code)
        return self.generate_answer(rag_prompt=rag_prompt, question=user_question)



