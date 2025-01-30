from lib.DataIngestion import DataIngestion
from lib.utilities import create_load_retriever
import os
from dotenv import load_dotenv
import argparse
load_dotenv()


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_path', type=str, default='/root/GenAI/.huggingface',
                        help='Document path')
    return parser.parse_args()


def initialize_retriever():
    elasticsearch_url = 'http://{elastic_username}:{elastic_password}@{elastic_host}:{elastic_port}'.format(
            elastic_username=os.environ['ELASTICSEARCH_USERNAME'],
            elastic_password=os.environ['ELASTICSEARCH_PASSWORD'],
            elastic_host=os.environ['ELASTICSEARCH_HOST'],
            elastic_port=os.environ['ELASTICSEARCH_PORT']
    )
    recreate = False if os.environ['RECREATE_INDEX'] == 'False' else True
    retriever = create_load_retriever(elasticsearch_url=elasticsearch_url,
                                      index_name=os.environ['ELASTICSEARCH_INDEX_NAME'], recreate=recreate,
                                      embedding_model_name=None)
    return retriever


def main():
    retriever = initialize_retriever()
    ingestor = DataIngestion(retriever=retriever, documents_path='parsed_azure/')
    ingestor.run()


if __name__ == "__main__":
    main()