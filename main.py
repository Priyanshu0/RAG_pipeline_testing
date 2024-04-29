import os
from dotenv import load_dotenv
import time
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
    
def load_reranker():
    return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def main(
    qa_dataset_path,
    data_path,
    test_llm_repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    rag_llm_repo_id="HuggingFaceH4/zephyr-7b-beta",
    num_docs_final = 4,
    num_docs_retrieved = 8,
    embedding_model_name = "text-embedding-ada-002",
    chunk_size=8191,
    threshold = 0.7
):
    load_credentials()
    start_time = time.time()
    # Initialize the RAG pipeline testing object
    llm_to_evaluate = llm_inference_api(rag_llm_repo_id)
    test_llm = llm_inference_api(test_llm_repo_id)
    reranker = load_reranker()
    deepeval_testing = RAG_pipeline_testing(qa_dataset_path, chunk_size, data_path, llm_to_evaluate=llm_to_evaluate, reranker = reranker,
                                            num_docs_final=num_docs_final, num_docs_retrieved=num_docs_retrieved, embedding_model_name=embedding_model_name, reuse=False)
    
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
    time_takn = end_time-start_time
    deepeval_results['total_time']=time_taken
    deepeval_results.to_csv("deep_eval_result.csv",index=False)

    llm_judge = LLM_Judge(repo_id=test_llm_repo_id, qa_dataset_path=qa_dataset_path,
                                            chunk_size=chunk_size, data_dir_path=data_path, llm_to_evaluate=llm_to_evaluate, reranker=reranker, embedding_model_name=embedding_model_name,
                                            num_docs_final=num_docs_final, num_docs_retrieved=num_docs_retrieved, reuse=False)
    
    llm_judge.generate_llm_eval_scores()
    llm_results = llm_judge.llm_eval_metrics
    llm_results["model_type"] = test_llm_repo_id
    llm_results["question_type"] = "synthetic"
    llm_results["total_cost"] = "$0"
    end_time2 = time.time()
    time_taken = end_time2 - end_time
    llm_results["total_time"] = time_taken

    return deepeval_results, llm_results

    


