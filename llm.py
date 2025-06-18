def get_proposal(model, tokenizer, prompt, model_name ='qwen'):
    #temperature=0.7, max_tokens=2048, seed=170, max_length=2048, truncation=True,do_sample=True, max_new_tokens=1024
    if model_name =='qwen':
        #messages = [ {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            #{"role": "user", "content": prompt}]
        messages = [ {"role": "system", "content": "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems:\nReason step by step\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."},
            {"role": "user", "content": prompt}]            
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512, temperature= 0.9, do_sample=True)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if isinstance(response, str) and '\n\n' in response:
            parts = re.split(r'\n\n', response)             
            if len(parts) > 2:
              response = parts[0] + '\n\n' + parts[1]
            else:
              response = parts[0]
              
            response += '\n\n'  
        if isinstance(response, str) and "Let's verify" in response:
            response = re.split(r"Let's verify", response)[0]
            response += '<end>'
        #response = re.split(r'aign', response)[0]

        return response

    elif model_name == 'vllm_qwen':

        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            n=1,
            logprobs=1,
            max_tokens=256,
            stop=['\n\n'],
        )

        output = model.generate(prompt, sampling_params, use_tqdm=False)      
        return output

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
