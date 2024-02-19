import json
import dashscope
from langchain.document_loaders import UnstructuredFileLoader
from Qwen import Qwen
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA






api_key=''

llm=Qwen(api_key=api_key)

filepath="E:\Langchain-CHAT\Langchain-Learning\data\\test.txt"


# 加载文件
#loader=TextLoader(filepath,encoding="utf-8")

loader=UnstructuredFileLoader(file_path=filepath)
docs=loader.load()

print(docs)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>文本分割<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#文件分割
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=200)
docs=text_splitter.split_documents(docs)
print(docs)


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>文本embedding<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


#构建向量数据库
model_name = "E:\ChatGLM3-6B\embedding\\bge-large-zh"
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,

)

db=FAISS.from_documents(docs,embedding)
# db=Chroma.from_documents(docs,embedding,persist_directory='VectoryStore')
# db.persist()
retriever=db.as_retriever()


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>开始对话<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

query = "3月份阿根廷会来中国吗？"


qa=RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=retriever)

response=qa.run(query)

print(response)




