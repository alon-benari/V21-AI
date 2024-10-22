import os

import sys
from langchain.document_loaders import WebBaseLoader, PyPDFLoader # load URLs.
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer

from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chromadb.utils import embedding_functions
import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
 )

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# openai.api_key = os.environ['OPENAI_API_KEY']

class PreProcess():
    '''
    A set of methods  to pre-process  text for a bot using the VDL library
    '''
    def __init__(self):
        self.docs = []
        #self.embedding = OpenAIEmbeddings()
        
        


    def concat_pdfdocs(self, pdfurl):
        '''
        A method to take in a PDF URL of a document and concatenate it to  to a larger document 
        '''

        # load a VDL PDF.
        loader  = PyPDFLoader(pdfurl)
        doc = loader.load()
        self.docs.extend(doc)
    
    def get_chuncks(self, chunck_size = 150, chunck_overlap = 10):
        '''
        A method to  create embedding following the splitting 
        '''
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunck_size,
            chunk_overlap = chunck_overlap,
            length_function=len,
            is_separator_regex=True,
            )
        splits = text_splitter.split_documents(self.docs)
        #
        
        return splits


    
    
    def delete_collection(self, collection, col_name):
        '''
        A method to delete a collection
        collection :
        '''
        collection.delete(name = col_name)


    def write_docs(self, list, docs2save = 'docs.txt'):
        '''
        A method to save the docs list to a flat text file for debug/ease of development
        '''

        with open(docs2save, 'w') as file:
            # Join the list elements into a single string with a newline character
            data_to_write = '**'.join(list.to_json()['kwargs']['page_content'])
    
             # Write the data to the file
            file.write(data_to_write)


    def load_chroma(self, splits):
        '''
        A method to take chunck  and load them into Chroma one by one using the
        '''
        get_page_content = lambda x: x.to_json()['kwargs']['page_content'].replace("\n","")
        get_metadata = lambda x: x.to_json()['kwargs']['metadata']
        get_id = lambda x: x.to_json()['kwargs']['id']
        #
        page_content = list(map(get_page_content, docs))
        metadata = list(map(get_metadata, docs))
        id = ['doc{0}'.format(i) for  i in range(len(page_content))]
        #
        # create a collection
        client = chromadb.PersistentClient(path="test")
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        #default_ef = SentenceTransformer("all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name="test", embedding_function = default_ef)
    #
        for i in range(len(id)):
            collection.add(
                documents=page_content[i],
                metadatas=metadata[i],
                ids=id[i]
            )
        return collection

    def connect_chroma(self,path):
        '''
        A method to commect to a chroma collection with a path "path"
        path - string, the path  name of collection
        '''
        client = chromadb.PersistentClient(path=path)
        #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        #default_ef = SentenceTransformer("all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name=path, embedding_function = default_ef)

        return collection

###########run#######
pp = PreProcess()


pdfurls = [
         "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiutm.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiu_util.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiuum.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiuhl7.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiuqr.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiu_1_250rn.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiu_1_0_297_ig.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiu_1_0_309_ig.pdf",
        "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiuim.pdf",
       "https://www.va.gov/vdl/documents/Clinical/CPRS-Text_Integration_Utility_(TIU)/tiuig.pdf"
]


for i, pdfurl in enumerate(pdfurls):
    pp.concat_pdfdocs(pdfurl)
docs = pp.docs

splits = pp.get_chuncks(chunck_size= 240, chunck_overlap=10)
# load into chromadb
#vs = pp.load_chroma(splits)
vs = pp.connect_chroma(path = "test") # connect to a collection

#simple query
def ask_q(question, n_res = 5):
    res = vs.query(
        query_texts=question,
        n_results = 5,
        include=["documents", "distances", "metadatas"]
    )

    # Define the prompt template for the LLM
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    # Create a chain combining the prompt template and LLM
    rag_chain = prompt | llm | StrOutputParser()

    doc_texts = "\\n".join([doc for doc in res['documents'][0]])

    answer = rag_chain.invoke({"question": question, "documents": doc_texts})

    return answer

question = ["How to create a new note"]
ans = ask_q(question)
# #
# def embed_chuncks(docs):
#     '''
#     A method to  create a vector store and embed the test docs
#     docs = list of documents of class Document
#     '''
    
#     embedding=OllamaEmbeddings(model="llama3", show_progress=True)
    
#     #
#     # create the vector store
    # #
    # vector_store = Chroma(
    #                     collection_name="lc_test", 
    #                     embedding_function=embedding, 
    #                     persist_directory= 'docs/lc_test'
    #                     )
    # #
    # get_page_content = lambda x: x.to_json()['kwargs']['page_content'].replace("\n","")
    # get_metadata = lambda x: x.to_json()['kwargs']['metadata']
    # get_id = lambda x: x.to_json()['kwargs']['id']
    # #
    # # get data
    # #
    # page_content = list(map(get_page_content, docs))
    # metadata = list(map(get_metadata, docs))
    # id = ['doc{0}'.format(i) for  i in range(len(page_content))]
    # #
    # # create Document list

    # documents =[
    #     Document( page_content = page_content[i],metadata = metadata[i],id = id[i])
    #             for i in range(len(id))
        
        
    # ]
    # uuids = [str(uuid4()) for _ in range(len(documents))]
    # vector_store.add_documents(documents = documents, ids = uuids)
    # # vector_store.delete_collection()
    # return vector_store


#
#

# q = "how to add a new note"
# persist_directory = 'docs/chroma/'
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")



# from langchain_chroma import Chroma

# vector_store = Chroma(
#     #collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory=persist_directory
# )

# # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma.from_documents(
#         documents=pp.docs,
#         embedding=embeddings,
#         persist_directory=persist_directory
# )
# response = vectordb.similarity_search(q,k=3)


# ### creating batches
# import chromadb
# from chromadb.utils.batch_utils import create_batches
# import uuid
# ### batching
# client = chromadb.PersistentClient(path="test-large-batch")
# large_batch = [(f"{uuid.uuid4()}", f"document {i}", [0.1] * 1536) for i in range(100000)]
# ids, documents, embeddings = zip(*large_batch)
# batches = create_batches(api=client,ids=list(ids), documents=list(documents), embeddings=list(embeddings))
# collection = client.get_or_create_collection("test")
# for batch in batches:
#     print(f"Adding batch of size {len(batch[0])}")
#     collection.add(ids=batch[0],
#                    documents=batch[3],
#                    embeddings=batch[1],
#                    metadatas=batch[2])