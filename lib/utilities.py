import os
from lib.CustomElasticSearch import CustomElasticSearch

def setup_chat_model(model_name, max_tokens=200):
    '''
    :param model_name: name of the LLM which we want to use to generate question and/or answers
    :return: Callable object used to invoke LLM
    '''

    if model_name == 'vllm':
        print(f"Using Mistral 7B")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key='EMPTY',
                    base_url=os.environ['VLLM_ENDPOINT'],
                     model=os.environ['VLLM_MODEL'],
                    temperature=0,
                    max_tokens=max_tokens)
    else:
        raise ValueError('Model not recognized')
    return llm


def create_load_retriever(elasticsearch_url, index_name, recreate: bool = False, embedding_model_name: str = None):
    """
    If recreate is true the index is deleted and recreated.
    Otherwise try to create the retriever and if it already exists load it instead.
    """
    print(f"Embedding model name: {embedding_model_name}")
    if recreate == True:
        retriever = CustomElasticSearch.create(elasticsearch_url=elasticsearch_url, index_name=index_name,
                                                      recreate=True, embedding_model_name=embedding_model_name)
    else:
        try:
            retriever = CustomElasticSearch.create(elasticsearch_url=elasticsearch_url,
                                                          index_name=index_name, recreate=False, embedding_model_name=embedding_model_name)  # Create index
        except Exception as e:
            print("Failed to create index. Error:", e)
            try:
                retriever = CustomElasticSearch(client=Elasticsearch(elasticsearch_url),
                                                       index_name=index_name, embedding_model_name=embedding_model_name)
            except Exception as e:
                print("Failed to create or load the retriever. Error:", e)
                return
    return retriever