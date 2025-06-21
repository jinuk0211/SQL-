import os
import argparse
import json
from tqdm import tqdm

import copy
import random
import numpy as np
from ordered_set import OrderedSet

random.seed(0)


def dump_json(obj, fname, indent=4, mode='w', encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def log_agent(agent, file_path):
    save_dict = agent
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'a') as f:
        json.dump(save_dict, f)
        f.write("\n")


parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--task_name", type=str, help="task_name", default="spider")  # spider
parser.add_argument("--input_file", type=str, help="Dev file", default="./")  # spider
parser.add_argument("--output_path", type=str, help="Dev file", default="")  # spider
# parser.add_argument("--split", type=int, help="split", default=0)
args = parser.parse_args()

para_configs = {
    "mcts_iters": 5,
    "deapth_limit": 20,
    "explore_rate": 100,
    "step_topk": 3,
    "reflect_threshold": 50.0,
    "reward_alpha": 0.4
}

def run_text2sql(model,tokenizer):
    llm_select = f'http://localhost:8000/llm'
    llm_simulate = f'http://localhost:8000/llm'
    llm_reward = f'http://localhost:8000/llm'
    base_model = {'select': llm_select, 'simulate': llm_simulate, 'reward': llm_reward}

    if args.task_name == "bird":
        # file_path = './dataset/SQL-o1_bird_dev_db_id_0.json'
        file_path = '/content/drive/MyDrive/text2sql/SQL-o1_spider_dev_db_id_0.json'
    # file_path = '/content/drive/MyDrive/text2sql/SQL-o1_spider_dev_db_id_0.json'
    file_path = '/content/drive/MyDrive/new/nw_input.json'
    sql_data = json.load(open(file_path))

    sql_data = sql_data[:3]
    # sql_data = json.load(open(f'/data/vda/dataset/0121_bird_dev_db_id_0_0_0_0.json'))
    # sql_data = json.load(open(f'/data/vda/dataset/0121_spider_dev_db_id_0_0_0_0.json'))

    save_sql_data = []
    os.makedirs('mcts_results', exist_ok=True)
    save_path = os.path.join('mcts_results', f'{args.task_name}_mcts_dev.json')

    # os.makedirs(f'/data/vda/mcts', exist_ok=True)
    # save_path = f'/data/vda/mcts/result/{args.task_name}/{args.task_name}_mcts_llama3-8b_2.json'

    prompt = para_configs.copy()

    for row in tqdm(sql_data):

# class AgentConfig(SearchConfig):
#     def __init__(self,
#                  base_model: LanguageModel,
#                  prompt: dict,
#                  llm,
#                  tokenizer,
#                  batch_size: int = 1,
#                  reward_alpha: float = 0.5,

        world_model = AgentWorldModel(base_model=base_model, prompt=prompt, max_steps=prompt['deapth_limit'])
        config = AgentConfig(base_model=base_model, prompt=prompt,llm=model,tokenizer=tokenizer, reward_alpha=prompt['reward_alpha'])
        algorithm = MCTS(depth_limit=prompt['deapth_limit'], disable_tqdm=False, output_trace_in_each_iter=True,
                         n_iters=prompt['mcts_iters'], w_exp=prompt['explore_rate'], cum_reward=np.mean, calc_q=max)  #
        reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
        result_rap = reasoner_rap(row)
        if row.get('target', ""):
            row['target'] = row['target'][:-1] if row['target'].endswith(';;') else row['target']

        # [(res[-1].state.blocks_state, res[-1].Q) for res in result_rap.trace_in_each_iter]
        # print("Answer:\n", row['target'])
        # for o in list(set([res[-1].state.blocks_state for res in result_rap.trace_in_each_iter])):
        #     print(o)

        # print(result_rap._output_cum_reward)
# result_rap = reasoner_rap(row)


        row['result_mcts'] = list(OrderedSet([( res[0], res[1][-1].state.blocks_state) for res in result_rap.trace_in_each_iter]))
        if result_rap.trace_worst[1]:
            row['result_mcts_worst'] = [(result_rap.trace_worst[0], result_rap.trace_worst[1][0][-1].blocks_state)]
        else:
            row['result_mcts_worst'] = ''

        if result_rap.trace[1]:
            row['result_mcts_best'] = [(result_rap.trace[0], result_rap.trace[1][0][-1].blocks_state)]
        else:
            row['result_mcts_best'] = ''

        save_sql_data.append(copy.deepcopy(row))
        dump_json(save_sql_data, save_path, indent=4)
#'mcts_results/{args.task_name}_mcts_dev.json'
VALUE_MODEL_DIR = "meta-llama/Llama-3.2-1B-Instruct"
# Qwen2.5-Coder-14B
tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    VALUE_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    # device_map="auto",  # Automatically choose best device
)
if __name__ == '__main__':
    run_text2sql(model,tokenizer)