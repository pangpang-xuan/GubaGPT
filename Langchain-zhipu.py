from langchain.text_splitter import CharacterTextSplitter
from ChatGLM3 import ChatGLM3
from langchain.document_loaders import UnstructuredFileLoader,DirectoryLoader
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer,AutoModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA



# def Pipei(query):
#     # 根据提问匹配上下文
#
#     docs = store.similarity_search(query)
#
#     context = [doc.page_content for doc in docs]
#     # print(type(context))
#     context = str(context)
#     print(context)
#     response, history = model.chat(tokenizer, context, history=[])
#     print(response)

model_path = "E:\ChatGLM3-6B\model"
llm = ChatGLM3()
llm.load_model(model_name_or_path=model_path)

# tokenizer = AutoTokenizer.from_pretrained("E:\ChatGLM3-6B\model", trust_remote_code=True)
# model = AutoModel.from_pretrained("E:\ChatGLM3-6B\model",trust_remote_code=True).cuda()
# model=model.eval()
# llm=HuggingFacePipeline(model=model,tokenizer=tokenizer)
filepath="E:\Langchain-CHAT\Langchain-Learning\data\\test.txt"


# 加载文件
#loader=UnstructuredFileLoader(filepath)
loader=TextLoader(filepath,encoding="utf-8")
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
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,

)
store=FAISS.from_documents(docs,embedding)

print(store)

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>开始对话<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

query = "梅西会来中国吗？"
#Pipei(query)

qa=RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=store.as_retriever())

response=qa.run(query)


# demo=gr.ChatInterface(chat)
#
# demo.launch(inbrowser=True)