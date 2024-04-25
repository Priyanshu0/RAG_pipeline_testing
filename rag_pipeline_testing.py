import pandas as pd
from typing import Optional
import pandas as pd
from pandas import DataFrame

from ragatouille import RAGPretrainedModel
from langchain_core.language_models.llms import LLM


from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric,
    BiasMetric,
    ToxicityMetric,
    GEval
)
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate

from rag_pipeline import RAG_pipeline

class RAG_pipeline_testing(RAG_pipeline):


    def __init__(self, qa_dataset_path: str, chunk_size: int, data_dir_path: str, llm_to_evaluate: LLM, num_docs_final: Optional[int] =2, 
                 reranker: Optional[RAGPretrainedModel] = None, num_docs_retrieved: Optional[int] = 5, qa_dataset: Optional[DataFrame] = None,
                 metrics_dataset_path: Optional[str] = None, reuse: Optional[bool] = True, embedding_model_name: Optional[str] = "thenlper/gte-small") -> None:
        super().__init__(data_dir_path= data_dir_path, chunk_size=chunk_size)
        if qa_dataset is not None:
            self.qa_dataset = qa_dataset
        else:
            self.qa_dataset = pd.read_csv(qa_dataset_path)
        self.llm = llm_to_evaluate
        self.knowledge_vector_database = super().load_embeddings(embedding_model_name=embedding_model_name, reuse=reuse)
        self.reranker = reranker
        self.num_docs_final = num_docs_final
        self.num_docs_retrieved = num_docs_retrieved
        if metrics_dataset_path is not None:
            self.deepeval_metrics_results = pd.read_csv(metrics_dataset_path)
        else:
            self.deepeval_metrics_results = None
        
        
    def create_golden_set(self, question_col_name: Optional[str] = "question", answer_col_name: Optional[str] = "answer") -> None:
        questions = self.qa_dataset[question_col_name].to_list()
        answers = self.qa_dataset[answer_col_name].to_list()
        golden_set = []
        for question, answer in zip(questions, answers):
            datapoint = {}
            datapoint["question"] = question
            datapoint["groundtruth"] = answer
            llm_answer, context = super().answer_with_rag(question, self.llm, self.knowledge_vector_database,
                                                           reranker=self.reranker, num_retrieved_docs=self.num_docs_retrieved,
                                                           num_docs_final=self.num_docs_final)
            final_prompt = self.RAG_PROMPT_TEMPLATE.format(question=question, context=context)
            datapoint["context"] = context
            datapoint["answer"] = self.format_llm_response(llm_answer)
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