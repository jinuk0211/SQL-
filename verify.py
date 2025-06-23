
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model caching
# VALUE_MODEL_DIR = "../hfmodels/Qwen/Qwen2.5-72B-Instruct"
VALUE_MODEL_DIR = "meta-llama/Llama-3.2-1B-Instruct"
global_value_model = None
global_tokenizer = None
from prompt import complete_query_from_subquery,complete_query_from_ans, complete_query_from_context
import numpy as np
import logging
import openai

from huggingface_hub import login

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model caching
# VALUE_MODEL_DIR = "../hfmodels/Qwen/Qwen2.5-72B-Instruct"
model = None
tokenizer = None
# from prompt import complete_query_from_subquery,complete_query_from_ans, complete_query_from_context
import numpy as np
import logging
import openai

# Load tokenizer


if torch.cuda.is_available():
    model = model.cuda()

def get_token_probabilities(model, tokenizer, text, idx, inputs=None):


    try:
        # Use pre-tokenized inputs if provided, otherwise tokenize text
        if inputs is None:
            inputs = tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        log_probs = []
        with torch.no_grad():
            # Get model outputs for the entire sequence
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits[0]  # Remove batch dimension

            # Calculate log probabilities for each position from idx
            for pos in range(
                idx - 1, input_ids.shape[1] - 1
            ):  # -1 because we predict next token
                # Get log probabilities for next token
                next_token_logits = logits[pos]
                log_probs_t = torch.log_softmax(next_token_logits, dim=-1)

                # Get log probability of the actual next token
                next_token_id = input_ids[0, pos + 1]
                log_prob = log_probs_t[next_token_id].item()
                log_probs.append(log_prob)

        return log_probs

    except Exception as e:
        print(f"Error calculating token probabilities: {str(e)}")
        return []



def get_query_token_probabilities(model,tokenizer,input, output):


    try:
        full_text = input + output
        inputs = tokenizer(
            full_text, truncation=True, return_tensors="pt"
        )

        input_tokens = tokenizer(
            input, padding=False, truncation=False, return_tensors="pt"
        )
        output_start_idx = input_tokens["input_ids"].shape[1]

        return get_token_probabilities(full_text, output_start_idx, inputs)

    except Exception as e:
        print(f"Error in get_query_token_probabilities: {str(e)}")
        return []

def probability(model,tokenizer,input, output, ans_weight=0.75):

    try:
        kl_dcp_probs = get_query_token_probabilities(model,tokenizer,input, output)
        if not kl_dcp_probs:
            return 0.0
        kl_dcp = -sum(kl_dcp_probs) / len(kl_dcp_probs)

        # 溫←츞?졿쓢亮녑쓦
        kl_loss = kl_dcp

        # ?졾컙??0,1]?븅뿴
        value = np.exp(-1.8 * (kl_loss - 1.8))
        value = 1 - (1 / (1 + value))

        return float(value)

    except Exception as e:
        logging.error(f"Error in risk value calculation: {str(e)}")
        return 0.0


# for subquestion in subquestions:
#   value = probability_select(user_question, subquestion, answer)
#   print(value)

import logging
from typing import List, Dict
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)
# value = mcts_task.get_step_value(child.y)
class AdvancedSelfConsistency:
    def __init__(self, client, model: str,  num_samples: int = 5, similarity_threshold: float = 0.8):
        self.client = client
        self.model = model
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
        self.self_consistency_completion_tokens = 0

    def generate_responses(self, system_prompt: str, user_prompt: str) -> List[str]:
        responses = []
        for _ in range(self.num_samples):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=4096
            )
            self.self_consistency_completion_tokens += response.usage.completion_tokens
            responses.append(response.choices[0].message.content)
        return responses

    def calculate_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def cluster_similar_responses(self, responses: List[str]) -> List[List[str]]:
        clusters = []
        for response in responses:  
            added_to_cluster = False
            for cluster in clusters:  
                if self.calculate_similarity(response, cluster[0]) >= self.similarity_threshold:
                    cluster.append(response)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
        return clusters

# rc = AdvancedSelfConsistency(similarity_threshold=0.9)
# responses = ['cat','dog','dog','lion','dog','dog lion cat','springclustering','springclusterin','dog','spring']
# print(rc.consistency_scores(responses))
# [0.1, 0.4, 0.4, 0.1, 0.4, 0.1, 0.2, 0.2, 0.4, 0.1]

    def consistency_scores(self, responses: List[str]) -> List[float]:
        n = len(responses)
        clusters = []
        response_to_cluster = {}

        for i, response in enumerate(responses):  
            added_to_cluster = False
            for cluster_id, cluster in enumerate(clusters):
                if self.calculate_similarity(response, cluster[0]) >= self.similarity_threshold:
                    cluster.append(response)
                    response_to_cluster[i] = cluster_id
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([response])
                response_to_cluster[i] = len(clusters) - 1

        # Map cluster sizes
        cluster_sizes = [len(c) for c in clusters]

        # Assign consistency score = cluster size / total
        scores = [cluster_sizes[response_to_cluster[i]] / n for i in range(n)]
        return scores

    def aggregate_results(self, responses: List[str]) -> Dict[str, any]:
        final_answers = responses
        clusters = self.cluster_similar_responses(final_answers)
        
        cluster_info = []
        for cluster in clusters:
            cluster_info.append({
                "answer": cluster[0],
                "frequency": len(cluster),
                "variants": cluster
            })
        
        cluster_info.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            "clusters": cluster_info,
            "total_responses": len(responses),
            "num_unique_clusters": len(clusters)
        }

    def evaluate(self, system_prompt: str, user_prompt: str) -> Dict[str, any]:
        responses = self.generate_responses(system_prompt, user_prompt)
        aggregated_result = self.aggregate_results(responses)
        
        return {
            "individual_responses": responses,
            "aggregated_result": aggregated_result
        }

def advanced_self_consistency_approach(system_prompt: str, initial_query: str, client, model: str) -> str:
    self_consistency = AdvancedSelfConsistency(client, model)
    result = self_consistency.evaluate(system_prompt, initial_query)
    
    logger.info("Advanced Self-Consistency Results:")
    logger.info(f"Total responses: {result['aggregated_result']['total_responses']}")
    logger.info(f"Number of unique clusters: {result['aggregated_result']['num_unique_clusters']}")
    for i, cluster in enumerate(result['aggregated_result']['clusters'], 1):
        logger.debug(f"\nCluster {i}:")
        logger.debug(f"  Representative answer: {cluster['answer']}")
        logger.debug(f"  Frequency: {cluster['frequency']}")
        logger.debug(f"  Variants: {cluster['variants']}")
    
    if result['aggregated_result']['clusters']:
        return result['aggregated_result']['clusters'][0]['answer'], self_consistency.self_consistency_completion_tokens
    else:
        return "No consistent answer found.", self_consistency.self_consistency_completion_tokens

# value = mcts_task.get_step_value(child.y)
# https://github.com/RyanLiu112/compute-optimal-tts/blob/main/src/reason/reranking/vote_utils.py
# def _agg_majority_vote(x_list: List[str], unused_v_list: List[float], return_reward=False):
#     counts = Counter(x_list)
#     most_common = max(counts, key=counts.get)
#     if return_reward:
#         return most_common, [0.0]
#     return most_common

# def get_step_value(self, actions):

