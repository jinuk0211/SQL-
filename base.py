from typing import NamedTuple
from collections import defaultdict
import sqlparse
import requests
import re
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from tqdm import trange
AgentAction = str
from base import *
from llm import get_proposal, llm_proposal
from verify import probability
CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'having', 'order by', 'limit', 'intersect', 'union', 'except', 'union all']
JOIN_KEYWORDS = ['join', 'on', 'as', 'right join', 'inner join', 'left join']
OTHER_KEYWORDS = ['distinct']
BIRD_KEYWORDS = ['if', 'else', 'datediff', 'over', 'instr', 'case', 'partition by', 'iif', 'float', 'real', 'when', 'int', 'using', 'timestampdiff', 'then', 'substr', 'cast', 'integer', 'strftime', 'end']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', 'not null', 'null']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(OTHER_KEYWORDS)
SQL_KEYWORDS.extend(BIRD_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)
SQL_KEYWORDS = [i.upper() for i in SQL_KEYWORDS]
from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from abc import ABC, abstractmethod
import numpy as np
from transformers import StoppingCriteriaList
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch
State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...
#------------------
    # def _expand(self, node: MCTSNode):

    #     if node.state is None:
    #         node.state = self.world_model.step(node.parent.state, node.action)
    #         node.reward, node.reward_details = self.search_config. \
    #             reward(node.parent.state, node.action, **node.fast_reward_details)
    #         node.is_terminal = self.world_model.is_terminal(node.state)

    #     if node.is_terminal:
    #         return

    #     children = []
    #     actions = self.search_config.get_actions(node.state)
#-------------------------

class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example, prompt = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example

class DefaultWorldModel(WorldModel):
    # A default implementation of WorldModel that only
    # saves the action sequence as the state

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def init_state(self):
        return []

    def step(self, state, action):
        return state + [action], {}

    def is_terminal(self, state):
        # By default the state is never terminal
        return False


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example

@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> AlgorithmOutput: ...

# reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
# result_rap = reasoner_rap(row)
class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo
# result_rap = reasoner_rap(row)

    def __call__(self, example: Example, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
    # def update_example(self, example: Example, prompt = None) -> None:
    #     if prompt is not None:
    #         self.prompt = prompt
    #     self.example = example
        self.search_config.update_example(example, prompt=prompt)
    # def update_example(self, example: Example, prompt = None) -> None:
    #     if prompt is not None:
    #       self.example = example
        return self.search_algo(self.world_model, self.search_config, **kwargs)


class AgentState(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: AgentAction

class AgentWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> AgentState:
        return AgentState(step_idx=0,
                          last_blocks_state="",
                          blocks_state="",
                          buffered_action="")
    # node.state = self.world_model.step(node.parent.state, node.action)
    def step(self, state: AgentState, action: AgentAction) -> tuple[AgentState, dict]:
        step_idx = state.step_idx
        # blocks_state = state.blocks_state + action + ("; " if action != "done" and action != "none" else "")

        if action == ";" or action == " ;" or action.endswith(";"):
            # blocks_state = state.blocks_state + ("" if state.blocks_state.endswith(";") or state.blocks_state.endswith("; ") else "; ") + action
            # blocks_state = state.blocks_state + " " + action
            blocks_state = state.blocks_state + action if not state.blocks_state else state.blocks_state + " " + action
        else:
            blocks_state = state.blocks_state + action if not state.blocks_state else state.blocks_state + " " + action

        new_buffered_action = action

        state = AgentState(step_idx=step_idx + 1,
                        last_blocks_state=state.blocks_state,
                        blocks_state=blocks_state,
                        buffered_action=new_buffered_action)
        return state

    def is_terminal(self, state: AgentState) -> bool:
        if state.buffered_action in [';', ' ;'] or state.buffered_action.endswith(";"):
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False



class AgentConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 llm,
                 tokenizer,
                 batch_size: int = 1,
                 reward_alpha: float = 0.5,
                 goal_reward_default: float = 0.,
                 goal_reached_reward: float = 100.) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.llm = llm
        self.tokenizer = tokenizer

    def lexical(self, query, values):
        if isinstance(query, str):
            for placeholder, value in values.items():
                query = query.replace(placeholder, value)
        elif isinstance(query, list):
            for i in range(len(query)):
                if query[i] in values:
                    query[i] = values[query[i]]
        return query

    def delexical(self, query):
        values = {}
        new_query = ""
        in_value = False
        in_col = False
        value = ""
        placeholder_id = 0
        new_query = ""
        for char in query:
            if char == "'":
                in_value = not in_value
                value += char
                if not in_value:
                    values[f"value_{placeholder_id}"] = value
                    new_query += f"value_{placeholder_id}"
                    placeholder_id += 1
                    value = ""
            else:
                if not in_value:
                    new_query += char
                else:
                    value += char
        return new_query, values

    def format_query(self, q, format_type):
        if format_type == 'unnormalized':
            return q["query"]
        elif format_type == 'normalized':
            return q["gold"]["query_normalized"]
        else:
            raise ValueError(f"format_type {format_type} not supported")

    def _is_whitespace(self, sqlparse_token):
        return sqlparse_token.ttype == sqlparse.tokens.Whitespace



    def normalize_sql(self, sql_exp):
        sql_exp = sql_exp.replace('"', "'")
        if sql_exp.count(
                "'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
            odd_quotes = True
        else:
            odd_quotes = False

        if not odd_quotes:
            sql_exp, values = self.delexical(sql_exp)
            sql_exp = sql_exp.lower()

        sql_exp = sql_exp.rstrip(";")
        parse = sqlparse.parse(sql_exp)
        sql = parse[0]
        flat_tokens = sql.flatten()
        sql_tokens = [
            (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
            for token in flat_tokens if not self._is_whitespace(token)
        ]

        sql_lower = ' '.join(sql_tokens)
        sql_lower = sql_lower.replace(' . ', '.')
        for op in AGG_OPS:
            sql_lower = sql_lower.replace(f" {op.upper()} (", f" {op.upper()}(")
        sql_lower = sql_lower.replace('( ', '(')
        sql_lower = sql_lower.replace(' )', ')')
        sql_lower = sql_lower.replace(' ,', ',')

        ### BIRD-SQL special cases ###
        sql_lower = sql_lower.replace(' AS text', ' AS TEXT')
        sql_lower = sql_lower.replace(' length(', ' LENGTH(')
        sql_lower = sql_lower.replace(' total(', ' TOTAL(')
        sql_lower = sql_lower.replace(' round(', ' ROUND(')
        ### END ###

        sql_lower = sql_lower.rstrip(";")
        sql_lower += ';'

        if not odd_quotes:
            # sql_tokens = self.lexical(sql_tokens, values)
            sql_lower = self.lexical(sql_lower, values)
        # else:
        #     print("Cannot process the following SQL")
        #     print(sql_exp, sql_tokens)
        return sql_lower

    # def segment_step(self, sql_completion):
    #     try:
    #         parse = sqlparse.parse(sql_completion)
    #         sql = parse[0]
    #     except Exception as e:
    #         return ""
    #     flat_tokens = sql.flatten()
    #     sql_tokens = [
    #         (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
    #         for token in flat_tokens
    #     ]

    #     step_length = 0
    #     for i, token in enumerate(sql_tokens[1:]):
    #         if token.lower() in CLAUSE_KEYWORDS:
    #             step_length = i + 1
    #             break

    #     if step_length == 0:
    #         # No more clauses, the entire completion is a step
    #         return sql_completion
    #     else:
    #         return "".join(sql_tokens[:step_length])
    def segment_step(sql_completion):
        try:
            parse = sqlparse.parse(sql_completion)
            sql = parse[0]
        except Exception as e:
            return ""
        flat_tokens = sql.flatten()
        sql_tokens = [
            (token.value.upper() if token.value in SQL_KEYWORDS else token.value)
            for token in flat_tokens
        ]

        step_length = 0
        clause_count = 0
        for i, token in enumerate(sql_tokens[1:]):
            if token.lower() in CLAUSE_KEYWORDS:
                print(token.lower())
                clause_count += 1
                if clause_count == 2:
                    step_length = i + 1
                break

        if clause_count == 0:
            print('clause 0개 감지')
            # No more clauses, the entire completion is a step
            print(sql_completion)
            return sql_completion
        elif clause_count == 1:
            print('clause 1개 감지')
            print(sql_completion)
            return sql_completion
        else:
            print('clause 2개 감지')
            print("".join(sql_tokens[:step_length]))
            return "".join(sql_tokens[:step_length])
    # SELECT product_name, price FROM products ORDER BY price DESC;


    def get_actions(self, state: AgentState) -> list[AgentAction]:
        if state.step_idx == self.prompt['deapth_limit']-1:
            if self.example['target'].startswith(state.blocks_state):
                return [('done',100.0)]
            else:
                return [('done',99.99)]
        else:
            print(state.blocks_state)
            print(self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state))
            input = self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state)
            outputs = []
            for i in range(4): #node 개수 actions args.noden
              output = llm_proposal(self.llm,self.tokenizer,input)
#def get_proposal(model, tokenizer, prompt, model_name ='qwen'):              

            def is_valid_string(s):
                if ";" not in s:
                    return False
                else:
                    return True

            # sql_completions = [key for key in output.keys() if is_valid_string(key)]
            # sql_completions = [self.normalize_sql(key) for key in output.keys() if is_valid_string(key)]
            sql_completions = [key for key in outputs if is_valid_string(key)]
            actions = set([
                (
                    self.segment_step(sql[len(state.blocks_state):].lstrip()).rstrip()
                    if len(sql) > len(state.blocks_state)
                    else sql
                )
                for sql in sql_completions
            ])

            actions = list(actions)
            p_reward = []
            for action in actions:
              reward = probability(self.llm,self.tokenizer,prompt,action)
#수정필요
# def probability(model,tokenizer,input, output, ans_weight=0.75):
              p_reward.append(reward)
            actions_scores_list = [(a,min(r,99.99)) for a,r in zip(actions, p_reward)]
            actions_scores_list = sorted(actions_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['step_topk']]
            return actions_scores_list

            # if self.example['output'].startswith(state.blocks_state):
            #     gt_action = self.example['output'][len(state.blocks_state):]
            #     actions_scores_list = [(gt_action, 100.0)]+[(a,r) for a,r in actions_scores_list if a!=gt_action]
                # actions_scores_list = [(gt_action, requests.post(self.base_model['select'], json={ "input": self.example['input']+state.blocks_state, "output": [gt_action] }).json()[0])]+[(a,r) for a,r in actions_scores_list if a!=gt_action]
            # p_reward = requests.post(self.base_model['select'], json={"input": self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state), "output": actions}).json()

    def fast_reward(self, state: AgentState, action: AgentAction) -> tuple[float, dict]:
        intuition = action[1]
        self_eval = intuition

        return (self.calculate_reward(intuition, self_eval),
                {'intuition': intuition, "self_eval": self_eval})

    def calculate_reward(self, intuition, goal_reached=None) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = goal_reached[1]
        else:
            goal_reward = goal_reached[1]
        return intuition * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: AgentState, action: AgentAction,
               intuition: float = None) -> tuple[float, dict]:
        # if action == "done" or action == "none" or action == " done":
        if action.endswith(";"):
            goal_reached_if = True
            goal_reached_score = requests.post(self.base_model['reward'], json={ "input":self.example['input'], "output": [state.blocks_state+action]}).json()[0]

            goal_reached = (goal_reached_if, goal_reached_score)
        else:
            goal_reached = (False, 0.0)
        return (self.calculate_reward(intuition, goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})





class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[MCTSNode]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.mean):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    # noinspection PyPep8Naming
    @property
    def Q(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)


class MCTSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_worst: Trace
    # trace_all: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None

class MCTSAggregation(Generic[State, Action, Example], ABC):
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode[State, Action,Example]) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action, Example]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])