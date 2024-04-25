import os
from dotenv import load_dotenv
load_dotenv()

from ragatouille import RAGPretrainedModel
from langchain_community.llms import HuggingFaceHub

from rag_pipeline import RAG_pipeline
from rag_pipeline_testing import RAG_pipeline_testing
from llm_judge import LLM_Judge


def load_credentials():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def llm_inference_api(repo_id="HuggingFaceH4/zephyr-7b-beta"):
    return HuggingFaceHub(
        repo_id=repo_id,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 2000,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )
    
def reranker():
    return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def main():
    
    


