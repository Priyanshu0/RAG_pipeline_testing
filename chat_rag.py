from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import time

pd.set_option("display.max_colwidth", None)
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

# Import the load_dotenv function from the dotenv module
from dotenv import load_dotenv

# Call the load_dotenv function to load environment variables from a .env file
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


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
    @st.cache_data
    def load_pdfs(_self, data_dir_path: str):
        loader = PyPDFDirectoryLoader(data_dir_path)
        docs = loader.load()
        _self.knowledge_base = [
            LangchainDocument(page_content=doc.page_content, metadata={"source": doc.metadata}) for doc in tqdm(docs)]
    @st.cache_data
    def split_documents(_self, tokenizer_name: str) -> List[LangchainDocument]:
        """
        Split documents into chunks of size `chunk_size` characters and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=_self.chunk_size,
            chunk_overlap=int(_self.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=_self.markdown_separators,
        )

        docs_processed = []
        for doc in _self.knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
    
        return docs_processed_unique
    @st.cache_data
    def load_embeddings(_self,
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
        _self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=False,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # set True to compute cosine similarity
        )

        # Check if embeddings already exist on disk
        index_name = f"index_chunk:{_self.chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
        index_folder_path = f"./data/indexes/{index_name}/"
        if os.path.isdir(index_folder_path) and reuse is True:
            return FAISS.load_local(
                index_folder_path,
                _self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True
            )

        else:
            print("Generating New Index")
            docs_processed = _self.split_documents(
                embedding_model_name,
            )
            knowledge_index = FAISS.from_documents(
                docs_processed, _self.embedding_model, distance_strategy=DistanceStrategy.COSINE
            )
            knowledge_index.save_local(index_folder_path)
            return knowledge_index
    @st.cache_data    
    def answer_with_rag(_self, question: str,
        _llm: LLM,
        _knowledge_index: VectorStore,
        reranker: Optional[RAGPretrainedModel] = None,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 7) -> Tuple[str, List[LangchainDocument]]:
        """Answer a question using RAG with the given knowledge index."""
        # Gather documents with retriever
        relevant_docs = _knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
        relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

        # Optionally rerank results
        if reranker:
            relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
            relevant_docs = [doc["content"] for doc in relevant_docs]

        relevant_docs = relevant_docs[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        final_prompt = _self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)

        # Redact an answer
        answer = _llm.invoke(final_prompt)

        return answer, relevant_docs
    
# rag_pipeline = RAG_pipeline(data_dir_path="./data", chunk_size=512)
# knowledge_vector_database = rag_pipeline.load_embeddings()
# print(knowledge_vector_database)
READER_LLM = OpenAI()
# response,con = rag_pipeline.answer_with_rag(question ='What is interest rate risk?', _llm= READER_LLM, _knowledge_index=knowledge_vector_database)
# print(con)

from langchain_community.llms import HuggingFaceHub

repo_id = "HuggingFaceH4/zephyr-7b-beta"
READER_MODEL_NAME = "zephyr-7b-beta"

ZEPHYR_LLM = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
READER_MODEL_NAME = "Mistral-7B-Instruct-v0.1"

MISTRAL_LLM = HuggingFaceHub(
    repo_id=repo_id,
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    SummarizationMetric,
    GEval
)
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
from pandas import DataFrame

class RAG_pipeline_testing(RAG_pipeline):


    def __init__(self, qa_dataset_path: str, chunk_size: int, data_dir_path: str, llm_to_evaluate: LLM, num_docs_final: Optional[int] =2, 
                 reranker: Optional[RAGPretrainedModel] = None, num_docs_retrieved: Optional[int] = 5, qa_dataset: Optional[DataFrame] = None,
                 metrics_dataset_path: Optional[str] = None, reuse: Optional[bool] = True) -> None:
        super().__init__(data_dir_path= data_dir_path, chunk_size=chunk_size)
        if qa_dataset is not None:
            self.qa_dataset = qa_dataset
        else:
            self.qa_dataset = pd.read_csv(qa_dataset_path)
        self.llm = llm_to_evaluate
        self.knowledge_vector_database = super().load_embeddings(reuse=reuse)
        self.reranker = None
        self._num_docs_final = num_docs_final
        self.num_docs_retrieved = num_docs_retrieved
        if metrics_dataset_path is not None:
            self.deepeval_metrics_results = pd.read_csv(metrics_dataset_path)
        else:
            self.deepeval_metrics_results = None
        
        
    def create_golden_set(self, question_col_name: Optional[str] = "question", answer_col_name: Optional[str] = "answer", context_col_name: Optional[str] = "context") -> None:
        questions = self.qa_dataset[question_col_name].to_list()
        answers = self.qa_dataset[answer_col_name].to_list()
        context = self.qa_dataset[context_col_name].to_list()
        golden_set = []
        for question, answer, context in zip(questions, answers,context):
            datapoint = {}
            datapoint["question"] = question
            datapoint["answer"] = answer
            datapoint["context"] = [context]
            llm_answer, context = super().answer_with_rag(question=question, _llm=self.llm,_knowledge_index= self.knowledge_vector_database,
                                                           reranker=self.reranker)
            final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
            datapoint["groundtruth"] = self.format_llm_response(llm_answer)
            datapoint["groundtruth"] = llm_answer
            datapoint["prompt"] = final_prompt
            golden_set.append(datapoint)
        self.golden_set = golden_set

    def format_llm_response(self, answer: str) -> None:
        fields = answer.split("<|assistant|>")
        if len(fields) == 1:
            return ""
        else:
            return fields[-1].strip("\n ")
        
    def deepeval_dataset(self) -> None:
        test_cases = []
        for datapoint in self.golden_set:
            if self.deepeval_metrics_results is not None and datapoint["question"] in self.deepeval_metrics_results["question"]:
                continue
            deepeval_test_case = LLMTestCase(input=datapoint["prompt"], actual_output=datapoint["answer"], expected_output=datapoint["groundtruth"],
            retrieval_context=datapoint["context"], context=datapoint["context"])
            test_cases.append(deepeval_test_case)
        self.dataset = EvaluationDataset(test_cases = test_cases)

    def deepeval_create_metrics(self, test_llm: LLM) -> None:
        coherence_metric = GEval(
                name="Coherence",
                evaluation_steps=["Check whether the sentences in 'actual output' aligns with that in 'expected output'"],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                model=test_llm
        )
        pii_metric = GEval(
            name="PII",
            evaluation_steps=["Check whether the 'actual output' contains any kind of personal information that makes a person identifiable"],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=test_llm
        )
        sentiment_metric = GEval(
            name="Positive Sentiment",
            evaluation_steps=["Check whether the 'actual output' has a positive tone or not"],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=test_llm
        )
        return [coherence_metric, pii_metric, sentiment_metric]


    def deepeval_metrics(self, test_llm: LLM, threshold: Optional[int] = 0.5) -> None:
        self.metrics = []
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=threshold, include_reason=True, model=test_llm)
        contextual_precision = ContextualPrecisionMetric(threshold=threshold, include_reason=True, model=test_llm)
        contextual_recall = ContextualRecallMetric(threshold=threshold, include_reason=True, model=test_llm)
        contextual_relevancy = ContextualRelevancyMetric(threshold=threshold, include_reason=True, model=test_llm)
        faithfulness = FaithfulnessMetric(threshold=threshold, include_reason=True, model=test_llm)
        bias_metric = BiasMetric(threshold=threshold, include_reason=True, model=test_llm)
        toxicity_metric = ToxicityMetric(threshold=threshold, include_reason=True, model=test_llm)
        custom_metrics = self.deepeval_create_metrics(test_llm=test_llm)
        self.metrics.extend([answer_relevancy_metric, contextual_precision,
                             contextual_recall, contextual_relevancy, faithfulness, bias_metric, toxicity_metric])
        self.metrics.extend(custom_metrics)
        self.results = self.dataset.evaluate(self.metrics)

    def format_results(self) -> None:
        all_datapoints = []
        for golden_datapoint, result in zip(self.golden_set, self.results):
            datapoint = {}
            datapoint["question"] = golden_datapoint["question"]
            datapoint["prompt"] = result.input
            datapoint["llm_answer"] = result.actual_output
            datapoint["groundtruth_answer"] = result.expected_output
            datapoint["retrieved_context"] = result.context
            datapoint["success"] = result.success
            for metric in result.metrics:
                metric_name = metric.__name__.replace(" ", "_").lower()
                datapoint[f"{metric_name}_score"] = metric.score
                datapoint[f"{metric_name}_success"] = metric.success
                datapoint[f"{metric_name}_reason"] = metric.reason
                datapoint[f"{metric_name}_evaluation_cost"] = metric.evaluation_cost
                datapoint["evaluation_model"] = metric.evaluation_model
            all_datapoints.append(datapoint)
        if self.deepeval_metrics_results is None:
            self.deepeval_metrics_results = pd.DataFrame(all_datapoints)
        else:
            self.deepeval_metrics_results = pd.concat([self.deepeval_metrics_results, pd.DataFrame(all_datapoints)], ignore_index=True)


# deepeval_testing = RAG_pipeline_testing("/Users/rahulkushwaha/Desktop/Gitllm/llm_eval/Eval result/qa.csv",512, "./data", READER_LLM, reranker=RERANKER)
# deepeval_testing.create_golden_set()
# print(deepeval_testing.golden_set)
# deepeval_testing.deepeval_dataset()
# deepeval_testing.dataset
# deepeval_testing.deepeval_metrics(test_llm="gpt-3.5-turbo", threshold=0.7)
# deepeval_testing.format_results()
# deepeval_results = deepeval_testing.deepeval_metrics_results


def evaluate_rag_pipeline(
    qa_dataset_path,
    chunk_size,
    data_path,
    reader_llm,
    reranker,
    test_llm,
    threshold,
):
    start_time = time.time()
    # Initialize the RAG pipeline testing object
    deepeval_testing = RAG_pipeline_testing(qa_dataset_path, chunk_size, data_path, reader_llm, reranker)
    
    # Create golden set
    deepeval_testing.create_golden_set()
    
    # Generate dataset
    deepeval_testing.deepeval_dataset()
     
    # Evaluate metrics
    deepeval_testing.deepeval_metrics(test_llm, threshold)
    
    # Format the results
    deepeval_testing.format_results()
    
    # Get the evaluation results
    deepeval_results = deepeval_testing.deepeval_metrics_results
    deepeval_results["model_type"] = "gpt-3.5-turbo"
    deepeval_results["question_type"] = "synthetic"
    end_time = time.time()
    time_take = end_time-start_time
    deepeval_results['total_time']=time_take
    deepeval_results.to_csv("deep_eval_result.csv",index=False)
    # Return the evaluation results
    return deepeval_results,time_take