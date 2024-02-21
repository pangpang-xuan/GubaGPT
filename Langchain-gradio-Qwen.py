import json

from langchain.document_loaders import UnstructuredFileLoader
from Qwen import Qwen
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr


def load_data(data_path):

    filepath=data_path

    loader = UnstructuredFileLoader(file_path=filepath)
    docs = loader.load()
    # 文件分割
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    return docs


def loadembedding(model_name):
    # 构建向量数据库
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,

    )
    return embedding


def chat(query,history):
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    response = qa.run(query)

    print(response)

    return response


api_key=''
filepath = "E:\Langchain-CHAT\Langchain-Learning\data\\test.txt" #文件路径
embedding_name="E:\ChatGLM3-6B\embedding\\bge-large-zh" #向量路径


docs=load_data(filepath) #加载数据
llm=Qwen(api_key=api_key) #加载模型
embedding=loadembedding(embedding_name) #加载embedding向量
db=FAISS.from_documents(docs,embedding)


retriever=db.as_retriever()



gr.ChatInterface(chat).launch(inbrowser=True)



