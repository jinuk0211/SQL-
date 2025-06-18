
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

def initialize_value_model():
    """Initialize the value model and tokenizer."""
    global global_value_model, global_tokenizer

    if global_value_model is not None and global_tokenizer is not None:
        return True  # Model already initialized

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_DIR)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            VALUE_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            # device_map="auto",  # Automatically choose best device
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        global_value_model = model
        global_tokenizer = tokenizer

        print("Value model initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing value model: {str(e)}")
        return False


def cleanup_value_model():
    """Cleanup model resources."""
    global global_value_model, global_tokenizer

    if global_value_model is not None:
        del global_value_model
        global_value_model = None

    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Value model resources cleaned up")


def get_token_probabilities(text, idx, inputs=None):
    """
    Calculate log probabilities for tokens from idx onwards.
    Each probability p(d_t|d_<t) is conditioned only on previous tokens.

    Args:
        text (str): Input text sequence d
        idx (int): Starting index for probability calculation
        inputs (dict, optional): Pre-tokenized inputs, if None will tokenize text

    Returns:
        list: List of log probabilities [log p(d_t|d_<t)] for t >= idx
    """
    global global_value_model, global_tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # Use pre-tokenized inputs if provided, otherwise tokenize text
        if inputs is None:
            inputs = global_tokenizer(
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
            outputs = global_value_model(
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



def get_query_token_probabilities(context, query):

    global global_value_model, tokenizer

    if (
        global_value_model is None or global_tokenizer is None
    ) and not initialize_value_model():
        return []

    try:
        # ?덂??답릉?뉑쑍?쉞okenization
        full_text = context + query
        inputs = global_tokenizer(
            full_text, truncation=True, return_tensors="pt"
        )

        # ?뺟떖野밧뎺?뉐걳tokenization竊뚧돻?컌uery?꾥돈冶뗤퐤營?
        context_tokens = global_tokenizer(
            context, padding=False, truncation=False, return_tensors="pt"
        )
        query_start_idx = context_tokens["input_ids"].shape[1]

        # ?룟룚query?ⓨ늽?꾣쫩?뉛펽鴉졾뀯藥쾢okenized?꼒nputs
        return get_token_probabilities(full_text, query_start_idx, inputs)

    except Exception as e:
        print(f"Error in get_query_token_probabilities: {str(e)}")
        return []


def probability_subquestion_question(ori_query, query, ans_weight=0.75):

    try:

        # 溫←츞decomposed query?▽뻑訝뗧쉪?잌쭓query礖귞럤
        kl_dcp_text_front = complete_query_from_subquery.format(query=query)
        kl_dcp_probs = get_query_token_probabilities(kl_dcp_text_front, ori_query)
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
        
def probability_context_question(ori_query, context, ans_weight=0.75): 

    try:

        # 溫←츞decomposed query?▽뻑訝뗧쉪?잌쭓query礖귞럤
        kl_dcp_text_front = complete_query_from_context.format(context=context)
        kl_dcp_probs = get_query_token_probabilities(kl_dcp_text_front, ori_query)
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
        
def probability_subanswer_question(ori_query, answer, ans_weight=0.75):

    try:
        # 溫←츞answer?▽뻑訝뗧쉪?잌쭓query礖귞럤
        kl_ans_text_front = complete_query_from_ans.format(answer=answer)
        kl_ans_probs = get_query_token_probabilities(kl_ans_text_front, ori_query)
        if not kl_ans_probs:
            return 0.0
        kl_ans = -sum(kl_ans_probs) / len(kl_ans_probs)


        # 溫←츞?졿쓢亮녑쓦
        kl_loss = kl_ans

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

def llm_proposal(model=None,tokenizer=None,prompt=None,model_name='qwen'):
    if model_name =='qwen':
        messages = [ {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    if model_name == 'gpt':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

    # elif model_name == 'llama':

    #     image = Image.open(img_path)

    #     messages = [
    #         {"role": "user", "content": [
    #             {"type": "image"},{"type": "text","text": f"{prompt}"}]}
    #             ]
    #     input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    #     inputs = processor(
    #         image,
    #         input_text,
    #         add_special_tokens=False,
    #         return_tensors="pt"
    #     ).to(model.device)

    #     output = model.generate(**inputs, max_new_tokens=512)
    #     output_text = processor.decode(output[0])
    #     split_text = output_text.split("<|end_header_id|>", 2)  # 理쒕? 2踰덈쭔 遺꾪븷

    #     # ??踰덉㎏ "<|end_header_id|>" ?댄썑 遺遺?媛?몄삤湲?(?덈떎硫?
    #     cleaned_text = split_text[2].strip()
    #     cleaned_text = cleaned_text.replace("<|eot_id|>", "")
    #     # print('get_proposal:理쒖쥌 ?띿뒪??')
    #     # print(cleaned_text)
    #     return cleaned_text

    #     # ?렞 異쒕젰 寃곌낵
    #     reply = response['choices'][0]['message']['content'].strip()
    #     return reply