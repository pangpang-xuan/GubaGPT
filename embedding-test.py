# from FlagEmbedding import FlagModel
#
# sentences = ["样例数据-1", "样例数据-2"]
# model = FlagModel('E:\ChatGLM3-6B\embedding\\bge-large-zh',
#                   use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# embeddings_1 = model.encode(sentences)
# print(embeddings_1)
# print(">>>>>>>>>>>>>>huggingface<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# from transformers import AutoTokenizer, AutoModel
# import torch
# # Sentences we want sentence embeddings for
#
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('E:\ChatGLM3-6B\embedding\\bge-large-zh')
# model = AutoModel.from_pretrained('E:\ChatGLM3-6B\embedding\\bge-large-zh')
# model.eval()
#
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
#
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#     # Perform pooling. In this case, cls pooling.
#     sentence_embeddings = model_output[0][:, 0]
# # normalize embeddings
# sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
# print(sentence_embeddings)
from langchain_community.vectorstores.faiss import FAISS

print("<<<<<<<<<<<<<<<<<<<<<<<langchain>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "E:\ChatGLM3-6B\embedding\\bge-large-zh"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,

)

docs=["dsfsdfsdfsdfsd", "sdssssssssssssssss"]
store=FAISS.from_documents(docs,model)

print(store)

