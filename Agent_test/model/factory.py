'''
提供所有模型实例
'''
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.chat_models.tongyi import ChatTongyi

from utils.config_handler import rag_conf

class BaseModelFFactory(ABC):

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:# chatmodel 和 embedding的父类
        pass



class ChatModelFactory(BaseModelFFactory):

    def generator(self) -> Optional[Embeddings | BaseChatModel]:# chatmodel 和 embedding的父类
        return ChatTongyi(
            model=rag_conf["chat_model_name"])
    

class EmbeddingModelFactory(BaseModelFFactory):

    def generator(self) -> Optional[Embeddings | BaseChatModel]:# chatmodel 和 embedding的父类
        return DashScopeEmbeddings(
            model=rag_conf["embedding_mode_name"])
    


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingModelFactory().generator()
