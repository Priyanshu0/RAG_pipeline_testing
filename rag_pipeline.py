from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import os
import shutil
import pandas as pd


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings

pd.set_option("display.max_colwidth", None)


class RAG_pipeline:

    def __init__(self, data_dir_path: str, chunk_size: int):
        self.data_dir_path = data_dir_path
        self.load_pdfs(self.data_dir_path)
        self.chunk_size = chunk_size
        self.RAG_PROMPT_TEMPLATE = """
            <|system|>
            Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.</s>
            <|user|>
            Context:
            {context}
            ---
            Now here is the question you need to answer.

            Question: {question}
            </s>
            <|assistant|>
        """
        self.markdown_separators = [
        "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            "",
        ]
    
    def load_pdfs(self, data_dir_path: str):
        loader = PyPDFDirectoryLoader(data_dir_path)
        docs = loader.load()
        self.knowledge_base = [
            LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata}) for doc in tqdm(docs)]

    def split_documents(self, tokenizer_name: str) -> List[LangchainDocument]:
        """
        Split documents into chunks of size `chunk_size` characters and return a list of documents.
        """
        if "text-embedding" not in tokenizer_name:
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                AutoTokenizer.from_pretrained(tokenizer_name),
                chunk_size=self.chunk_size,
                chunk_overlap=int(self.chunk_size / 10),
                add_start_index=True,
                strip_whitespace=True,
                separators=self.markdown_separators,
            )
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=int(self.chunk_size / 10),
                add_start_index=True,
                strip_whitespace=True,
                separators=self.markdown_separators,
            )

        docs_processed = []
        for doc in self.knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
    
        return docs_processed_unique
    
    def load_embeddings(self,
        embedding_model_name: Optional[str] = "thenlper/gte-small", reuse: Optional[bool] = True) -> FAISS:
        """
        Creates a FAISS index from the given embedding model and documents. Loads the index directly if it already exists.

        Args:
            langchain_docs: list of documents
            chunk_size: size of the chunks to split the documents into
            embedding_model_name: name of the embedding model to use

        Returns:
            FAISS index
        """
        # load embedding_model
        if "text-embedding" not in embedding_model_name:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                multi_process=False,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},  # set True to compute cosine similarity
            )
        else:
            self.embedding_model = OpenAIEmbeddings(
                model=embedding_model_name
            )

        # Check if embeddings already exist on disk
        index_name = f"index_chunk:{self.chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
        index_folder_path = f"./data/indexes/{index_name}/"
        if os.path.isdir(index_folder_path) and reuse is True:
            return FAISS.load_local(
                index_folder_path,
                self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=False
            )

        else:
            print("Generating New Index")
            docs_processed = self.split_documents(
                embedding_model_name,
            )
            knowledge_index = FAISS.from_documents(
                docs_processed, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            if os.path.isdir(index_folder_path):
                shutil.rmtree(index_folder_path)
            knowledge_index.save_local(index_folder_path)
            return knowledge_index
        
    def answer_with_rag(self, question: str,
        llm: LLM,
        knowledge_index: VectorStore,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 7) -> Tuple[str, List[LangchainDocument]]:
        """Answer a question using RAG with the given knowledge index."""
        # Gather documents with retriever
        relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
        relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

        # Optionally rerank results
        if reranker:
            relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        # Redact an answer
        answer = llm.invoke(final_prompt)

        return answer, relevant_docs