# GudaGPT

## 本项目实现了从股吧某指定网址爬取实时股吧帖子信息，写入本地文件，再使用`langchain`进行本地知识库的检索。
### `由于某种原因，本项目暂不提供爬虫部分代码（请自行实现）`
## 环境准备
```
$ python --version
Python 3.10
```
```shell
# 拉取仓库
git clone https://github.com/jayofhust/GudaGPT.git
```
```shell
# 安装全部依赖
pip install -r requirements.txt 
```
## 模型准备
+ 本项目中使用的 LLM 模型 [THUDM/ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) 
+ Embedding 模型 [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)
+ 自动下载模型需要先[安装 Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行
```Shell
$ git lfs install
$ git clone https://huggingface.co/THUDM/chatglm3-6b
$ git clone https://huggingface.co/BAAI/bge-large-zh
```
+ 也可以点击链接手动下载模型

## 网页端（Langchain-gradio-Qwen.py）
根据实际修改下面路径
+ api_key=''
+ filepath = "E:\Langchain-CHAT\Langchain-Learning\data\\test.txt" #文件路径
+ embedding_name="E:\ChatGLM3-6B\embedding\\bge-large-zh" #向量路径
```Shell
python Langchain-gradio-Qwen.py
```

## 调用本地大模型版本 （Langchain-zhipu.py）
+ 把model_path和filepath和model_name切换成本地的模型路径，文件路径，embedding模型路径
+ 本项目使用的是FAISS向量数据库，进行query与vector的匹配
+ query可以进行更换
```shell
python Langchain-zhipu.py
```
## 调用qwen（遵义千问）-api版本 （Langchain-qwen.py）
+ 把api_key和model_name切换成自己的api-key，embedding模型路径
+ 本项目使用的是FAISS向量数据库，进行query与vector的匹配
+ query可以进行更换
```shell
python Langchain-qwen.py
```