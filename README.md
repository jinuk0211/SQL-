# SQL-
```python
class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter']
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)


        if not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            self._simulate(path)


        # while not self._is_terminal_with_depth_limit(path[-1]):
        #     self._expand(path[-1])
        #     # ### debug mode
        #     # if path[-1].parent is not None:
        #     #     self._back_propagate(path)
        #     if self._is_terminal_with_depth_limit(path[-1]) or len(path[-1].children) == 0:
        #         break
        #     fast_rewards = [child.fast_reward for child in path[-1].children]
        #     node = path[-1].children[self.simulate_choice(fast_rewards)]
        #     path.append(node)

        cum_reward = self._back_propagate(path)
        if self.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > self._output_cum_reward:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_iter':
            self._output_cum_reward = cum_reward
            self._output_iter = path
        if self.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
            self._output_cum_reward = cum_reward
            self._output_iter = path
        return cum_reward, path

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._uct_select(node)

    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            expl = [c for c in node.children if c.fast_reward_details['intuition']!=100.0] # expl = [c for c in node.children]
            if any([len(c.cum_rewards)>0 for c in expl]):
                # return max([c for c in node.children if c.fast_reward_details['intuition']==100.0], key=self._uct)
                return max([c for c in node.children], key=self._uct)
            else:
                return max([c for c in node.children if c.fast_reward_details['intuition']==100.0 or len(c.cum_rewards)==0], key=self._uct)
        else:
            unvisited_children = filter(lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.fast_reward)

    def _expand(self, node: MCTSNode):
             
        if node.state is None:
            node.state = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config. \
                reward(node.parent.state, node.action, **node.fast_reward_details)
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        # print(f'Step {node.state.step_idx + 1}: ')
        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            fast_reward, fast_reward_details = action[1], {'intuition': action[1]}
            # print(action[0])
            # print(fast_reward)
            child = MCTSNode(state=None, action=action[0], parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q)
            children.append(child)
        # print()

        node.children = children

    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]):
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])


    def _dfs_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return (self.cum_reward([node.reward for node in path[1:]]), path)
        if cur.children is None:
            return (-math.inf, path)
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return (-math.inf, path)
        return [self._dfs_max_reward(path + [child]) for child in visited_children]

    def _dfs_min_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return min((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, calc_q=self.calc_q)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            cum_reward, path = self.iterate(self.root)
            if self.output_trace_in_each_iter:
                # self.trace_in_each_iter.append(deepcopy(path))
                self.trace_in_each_iter.append(deepcopy((cum_reward, path)))

        if self.output_strategy == 'follow_max':
            self._output_iter = []
            cur = self.root
            while True:
                self._output_iter.append(cur)
                if cur.is_terminal:
                    break
                visited_children = [x for x in cur.children if x.state is not None]
                if len(visited_children) == 0:
                    break
                cur = max(visited_children, key=lambda x: x.reward)
            self._output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])
        if self.output_strategy == 'max_reward':
            self._output_cum_reward, self._output_iter = self._dfs_max_reward([self.root])
            self._output_cum_reward_worst, self._output_iter_worst = self._dfs_min_reward([self.root])
            # self._output_iter_all = self._dfs_reward([self.root])


            if self._output_cum_reward == -math.inf:
                self._output_iter = None

            if self._output_cum_reward_worst == -math.inf:
                self._output_iter_worst = None

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 log_file: Optional[str] = None,
                 **kwargs) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter_worst is None:
            terminal_state_worst = trace_worst = None
        else:
            terminal_state_worst = self._output_iter_worst[-1].state
            trace_worst = [node.state for node in self._output_iter_worst], [node.action[0] for node in self._output_iter_worst[1:]]

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action[0] for node in self._output_iter[1:]]
            
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=(self._output_cum_reward, trace),
                            trace_worst=(self._output_cum_reward_worst, trace_worst),
                            # trace_all=self._output_iter_all,
                            trace_of_nodes=self._output_iter,
                            tree_state=self.root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_worst=result.trace_worst,
                # trace_all=result._output_iter_all,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from collections import defaultdict
import numpy as np
from tqdm import trange

from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace


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

```

```python

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

class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Example, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)
```
```python
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

    def segment_step(self, sql_completion):
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
        for i, token in enumerate(sql_tokens[1:]):
            if token.lower() in CLAUSE_KEYWORDS:
                step_length = i + 1
                break

        if step_length == 0:
            # No more clauses, the entire completion is a step
            return sql_completion
        else:
            return "".join(sql_tokens[:step_length])

    def get_actions(self, state: AgentState) -> list[AgentAction]:
        if state.step_idx == self.prompt['deapth_limit']-1:
            if self.example['target'].startswith(state.blocks_state):
                return [('done',100.0)]
            else:
                return [('done',99.99)]

            # if self.example['output'].startswith(state.blocks_state):
            #     return [('done',100.0)]
            # else:
            #     return [('done',99.99)]
        else:
            # output = requests.post(self.base_model['select'], json={"instruction": self.example['instruction'], "input": self.example['instruction'] + "\n" +self.example['input']+state.blocks_state, "output": [] }).json()
            # print(self.example['input'])
            print(state.blocks_state)
            print(self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state))
            # input()
            output = requests.post(self.base_model['select'], json={ "input": self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state), "output": [] }).json()

            # def is_valid_string(s):
            #     pattern = r'^(\[[^\]]+\]: <[^>]+>)'
            #     if "; " not in s:
            #         return bool(s in ['none','done'])
            #     else:
            #         if not s.endswith("; done"):
            #             return False
            #         else:
            #             #  and x.split('<')[-1].split('>')[0] in self.example['input']
            #             return all([bool(re.match(pattern, x)) for x in s.split("; ")[:-1]])

            # def is_valid_string(s):
            #     if ";" not in s:
            #         if s == "done" or s == " done":
            #             return s
            #         return ""
            #     else:
            #         if s == "; done" or s == ";done":
            #             return "; done"
            #         elif s.endswith("done"):
            #             return s.split("done")[0]
            #         else:
            #             return s

            # sql_completions = []
            # for key in output.keys():
            #     key = is_valid_string(key)
            #     if key:
            #         if key not in ["done", " done", "; done", ";done"]:
            #             sql_completions.append(self.normalize_sql(key))
            #         else:
            #             sql_completions.append(key)
            #     else:
            #         continue

            def is_valid_string(s):
                if ";" not in s:
                    return False
                else:
                    return True

            sql_completions = [key for key in output.keys() if is_valid_string(key)]
            # sql_completions = [self.normalize_sql(key) for key in output.keys() if is_valid_string(key)]

            actions = set([
                (
                    self.segment_step(sql[len(state.blocks_state):].lstrip()).rstrip()
                    if len(sql) > len(state.blocks_state)
                    else sql
                )
                for sql in sql_completions
            ])

            actions = list(actions)

            # p_reward = requests.post(self.base_model['select'], json={"input": self.example['instruction'] + "\n" + self.example['input']+state.blocks_state, "output": actions}).json()

            p_reward = requests.post(self.base_model['select'], json={"input": self.example['input'].replace("The incomplete SQL query:\n", "The incomplete SQL query:\n" + state.blocks_state), "output": actions}).json()
            actions_scores_list = [(a,min(r,99.99)) for a,r in zip(actions, p_reward)]
            actions_scores_list = sorted(actions_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['step_topk']]
            
            # if self.example['output'].startswith(state.blocks_state):
            #     gt_action = self.example['output'][len(state.blocks_state):]
            #     actions_scores_list = [(gt_action, 100.0)]+[(a,r) for a,r in actions_scores_list if a!=gt_action]
                # actions_scores_list = [(gt_action, requests.post(self.base_model['select'], json={ "input": self.example['input']+state.blocks_state, "output": [gt_action] }).json()[0])]+[(a,r) for a,r in actions_scores_list if a!=gt_action]        
            return actions_scores_list

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
            # goal_reached_score = requests.post(self.base_model['reward'], json={ "input": self.example['instruction'] + "\n" + self.example['input'], "output": [state.blocks_state+action]}).json()[0]
            goal_reached_score = requests.post(self.base_model['reward'], json={ "input":self.example['input'], "output": [state.blocks_state+action]}).json()[0]

            goal_reached = (goal_reached_if, goal_reached_score)
        else:
            goal_reached = (False, 0.0)
        return (self.calculate_reward(intuition, goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})

from reasoners.visualization import visualize,visualize_save,visualize_out
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode

# (Optional) You can write node_data_factory and edge_data_factory to show customized information.
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else "Not expanded",
                    #  "function state": '\n'.join(n.state.functions_state) if n.state else "Not expanded",
                    #  "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
                     "Q": n.Q,
                     "intuition": n.fast_reward_details["intuition"] if n.id!=0 else "N/A",
                     "# visited": len(n.cum_rewards)})

def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    return EdgeData({# "Q": n.Q,
                    #  "intuition": n.fast_reward_details["intuition"],
                    #  "self_eval": n.fast_reward_details["intuition"],
                     "action": n.action})
    
def visualize_mcts(result_rap):
    visualize(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)   
    
def visualize_mcts_save(result_rap):
    return visualize_save(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)  
    
def visualize_mcts_out(data):
    visualize_out(data) 
```

```python
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import re
import sqlite3
import traceback
import os
from vllm import LLM, SamplingParams
from func_timeout import func_set_timeout
import func_timeout
import tqdm

prompt_cw_temp_sft = """Given the following database schema and question, your task is to write a valid SQL query whose execution will accurately answer the question. If the value below the incomplete SQL query is not empty, your task is to complete it into a full SQL query. Remember to end the query with a semicolom ```;```.

Database schema:
{ds}

Sample rows of each table:
{sr}

Question:
{qs}{hint}

Question hint:
{sql}

The incomplete SQL query:
{sql}

Answer the question by a SQL query only with no explanation:
"""


def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print("=" * 10, e)
        return None


class LLM_Model(object):
    def __init__(self, model=''):

        self.model = model
        model = model.lower().replace('_', '').replace('-', '')
        if 'qwen2' in model:
            self.tag = 'qwen2'
        elif 'llama3' in model:
            self.tag = 'llama3'
        elif 'llama2' in model:
            self.tag = 'llam2'
        elif 'deepseek' in model:
            self.tag = 'deepseek'
        elif 'mistral' in model:
            self.tag = 'mistral'
        elif 'codellama' in model:
            self.tag = 'codellama'
        else:
            raise TypeError(f"Unexpect model: {model}.")

        self.llm = LLM(model=self.model,
                       seed=123,
                       tensor_parallel_size=args.gpus,
                       trust_remote_code=True,
                       gpu_memory_utilization=0.9
                       )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_response(self, prompts, max_tokens=1024, temperature=0.01, top_p=0.5):
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                         skip_special_tokens=True, stop=self.tokenizer.eos_token)
        if self.tag in ['mistral']:
            messages_list = [[{"role": "user", "content": p}] for p in prompts]
        else:
            messages_list = [
                [{"role": "system", "content": "You are a helpful SQLite assistant."}, {"role": "user", "content": p}]
                for p in prompts]
        messages_list = self.tokenizer.apply_chat_template(messages_list, add_generation_prompt=True, tokenize=False)
        outputs = self.llm.generate(messages_list, sampling_params)
        return [output.outputs[0].text for output in outputs]


class LLM_Online(object):
    def __init__(self, model="qwen72b", device=[0]):
        None

    def generate_response(self, prompts):
        rs = []
        for prompt in tqdm.tqdm(prompts):
            res = None  # your online LLM
            rs.append(res)
        return rs


def parse_dataset(data_path, mode='dev', dataset='bird'):
    # redirect path
    data_tuples_path = ''
    if dataset == 'bird':
        # data_tuples_path = os.path.join(data_path, dataset, mode, f'{mode}.json')
        data_tuples_path = os.path.join(data_path, dataset, mode, f'{mode}.json')
    elif 'spider_DK' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'Spider_DK.json')
    elif 'spider_real' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'spider-realistic.json')

    elif 'spider_syn' == dataset:
        data_tuples_path = os.path.join(data_path, 'spider', 'dev.json')
    elif 'spider' in dataset:
        if mode == 'test':
            data_tuples_path = os.path.join(data_path, 'spider', 'test.json')
        else:
            data_tuples_path = os.path.join(data_path, 'spider', f'{mode}.json')
    else:
        raise TypeError(f"Unexpect dataset: {dataset}.")

    data_tuples = read_json_file(data_tuples_path)

    return data_tuples


def convert_fk_index(data):
    fk_holder = []
    table_names_original = [i.lower() for i in data['table_names_original']]  # some bug
    column_names_original = [(i[0], i[1].lower()) for i in data['column_names_original']]
    for fk in data["foreign_keys"]:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        if type(tn) is str:
            tn = tn.lower()
        if type(col) is str:
            col = col.lower()
        if type(ref_tn) is str:
            ref_tn = ref_tn.lower()
        if type(ref_col) is str:
            ref_col = ref_col.lower()
        ref_cid, cid = None, None
        try:
            tid = table_names_original.index(tn)
            ref_tid = table_names_original.index(ref_tn)
            for i, (tab_id, col_org) in enumerate(column_names_original):
                if tab_id == ref_tid and ref_col == col_org:
                    ref_cid = i
                elif tid == tab_id and col == col_org:
                    cid = i
            if ref_cid and cid:
                fk_holder.append([cid, ref_cid])
        except:
            traceback.print_exc()
            print("table_names_original: ", table_names_original)
            print("finding tab name: ", tn, ref_tn)
            print(data)
            # sys.exit()
    return fk_holder


def dump_db_json_schema(db, f):
    '''read table and column info'''

    try:
        conn = sqlite3.connect(db)
    except:
        print(db)
        exit()
    conn.execute('pragma foreign_keys=ON')
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {'db_id': f,
            'table_names_original': [],
            'table_names': [],
            'column_names_original': [(-1, '*')],
            'column_names': [(-1, '*')],
            'column_types': ['text'],
            'primary_keys': [],
            'foreign_keys': []}

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace("_", ' '))
        fks = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data['column_names_original'].append((i, col[1]))
            data['column_names'].append((i, col[1].lower().replace("_", " ")))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if 'char' in col_type or col_type == '' or 'text' in col_type or 'var' in col_type:
                data['column_types'].append('text')
            elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type or 'number' in col_type \
                    or 'id' in col_type or 'real' in col_type or 'double' in col_type or 'float' in col_type:
                data['column_types'].append('number')
            elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                data['column_types'].append('time')
            elif 'boolean' in col_type:
                data['column_types'].append('boolean')
            else:
                data['column_types'].append('others')

            if col[5] == 1:
                data['primary_keys'].append(len(data['column_names']) - 1)

    data["foreign_keys"] = fk_holder
    data['foreign_keys'] = convert_fk_index(data)

    return data


def get_schema_dict(db, kk=3):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """
    data = dump_db_json_schema(db, db.split('/')[-1])
    tables = data['table_names_original']
    column_types = data['column_types']
    primary_keys = data['primary_keys']
    foreign_keys = data['foreign_keys']
    column_names = data['column_names_original']

    schema_dict = {
        'tables': {},
        'foreign_keys': []
    }

    for i, table in enumerate(tables):
        t = {}
        for j, c in enumerate(column_names):
            if c[0] == i:
                if j in primary_keys:
                    t[c[1]] = [column_types[j].upper(), True]
                else:
                    t[c[1]] = [column_types[j].upper(), True]
        schema_dict['tables'][table] = t

    for foreign_key in foreign_keys:
        t1 = tables[column_names[foreign_key[0]][0]]
        c1 = column_names[foreign_key[0]][1]
        t2 = tables[column_names[foreign_key[1]][0]]
        c2 = column_names[foreign_key[1]][1]
        schema_dict['foreign_keys'].append([t1, c1, t2, c2])

    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    # get exapmles
    for table in schema_dict['tables'].keys():
        try:
            select_query = f'SELECT * FROM `{table}` LIMIT {kk}'
            cursor.execute(select_query)
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info(`{table}`);")
            columns = [column[1] for column in cursor.fetchall()]
            for i, c in enumerate(columns):
                cls_valuse = [f"{row[i][0:100]}..." if type(row[i]) is str and len(row[i]) > 100 else row[i] for row in
                              rows]
                schema_dict['tables'][table][c].append(cls_valuse)
        except Exception as e:
            print(e)
    return schema_dict


def get_example_str(schema_dict, k=1):
    tables = list(schema_dict['tables'].keys())
    examples = {}
    for table in tables:
        table_dict = schema_dict['tables'][table]
        example = []
        for cls in table_dict.keys():
            example.append(table_dict[cls][2])
        example_str = []
        for i, v in enumerate(example[0]):
            example_str.append(tuple([e[i] for e in example]))
            if (i + 1) == k:
                break
        examples[table] = example_str

    e_s = ''
    for key in examples.keys():
        e_s += f"{key}: " + str(examples[key]) + '\n'

    return e_s[:-1]


def get_schmea_str_and_examples(schema_dict):
    schmea_str = ""
    tables = list(schema_dict['tables'].keys())
    examples = {}
    for table in tables:
        if ' ' in table:
            table_str = f'CREATE TABLE "{table}" ('
        else:
            table_str = f"CREATE TABLE {table} ("
        table_dict = schema_dict['tables'][table]

        pk_str = ''
        example = []
        for cls in table_dict.keys():
            try:
                cls_ = f'"{cls}"' if ' ' in cls else cls
                table_str += f"{cls_} {table_dict[cls][0]}, "
                if table_dict[cls][1]:
                    pk_str += cls_ + ', '
                example.append(table_dict[cls][2])
            except Exception as e:
                print(e)
        example_str = []

        try:
            for i, v in enumerate(example[0]):
                example_str.append(tuple([e[i] for e in example]))
        except Exception as e:
            print(e)

        examples[table] = example_str

        if pk_str != '':
            table_str += f"PRIMARY KEY({pk_str[:-2]}), "

        fk_str = ''
        for fk in schema_dict['foreign_keys']:
            if fk[0] == table and fk[2] in tables:
                if fk[3] in schema_dict['tables'][fk[2]].keys():
                    fk = [f'"{f}"' if ' ' in f else f for f in fk]
                    fk_str += f'FOREIGN KEY ({fk[1]}) REFERENCES {fk[2]}({fk[3]}), '
        if fk_str != '':
            table_str += fk_str

        schmea_str += table_str[:-2] + '); '

    schmea_str = schmea_str[:-1]

    e_s = ''
    for key in examples.keys():
        e_s += f"{key}: " + str(examples[key]) + '\n'

    return schmea_str, e_s[:-1]


# parse SQL
def parse_sql_from_string(input_string):
    input_string = input_string.replace('\n', ' ').replace('\t', '')
    rs = ''
    if '```sql' in input_string:
        try:
            sql_pattern = r'```sql(.*?)```'
            all_sqls = []
            for match in re.finditer(sql_pattern, input_string, re.DOTALL):
                all_sqls.append(match.group(1).strip())
            if all_sqls:
                rs = all_sqls[-1]
                if 'SELECT' not in rs and len(all_sqls) > 1:
                    rs = all_sqls[-2]
        except:
            None
    if 'select' in input_string.lower() and rs == '':
        rs = input_string[input_string.find('SELECT'):]
    if ';' in rs:  # end
        rs = rs[:input_string.find(';') + 1]
    if rs == '':
        rs = 'SELECT xx FROM xx'
    return replace_multiple_spaces(rs).replace('```', '')


def replace_multiple_spaces(text):
    return re.sub(r'\s{2,}', ' ', text)


def filter_dict_by_sql(schema_dict, sql):
    schema_dict_ = copy.deepcopy(schema_dict)
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    # tables
    for table in keys:
        if f'from {table.lower()}' not in sql.lower() and f'join {table.lower()}' not in sql.lower():
            schema_dict_['tables'].pop(table, None)
    # columns
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    for table in keys:
        cls_keys = list(schema_dict_['tables'][table].keys())
        cls_keys.sort(key=lambda x: - len(x))
        tabel_dict = copy.deepcopy(schema_dict_['tables'][table])
        for cls in cls_keys:
            if cls.lower() not in sql.lower():
                schema_dict_['tables'][table].pop(cls, None)
        if len(schema_dict_['tables'][table].keys()) == 0:
            # schema_dict_['tables'][table] = tabel_dict  # for COUNT(*)
            for cls in tabel_dict.keys():
                if tabel_dict[cls][1] == True:
                    schema_dict_['tables'][table][cls] = tabel_dict[cls]

        if len(schema_dict_['tables'][table].keys()) == 0:
            schema_dict_['tables'][table][tabel_dict.keys()[0]] = tabel_dict[tabel_dict.keys()[0]]
            schema_dict_['tables'][table][tabel_dict.keys()[1]] = tabel_dict[tabel_dict.keys()[1]]
            # for COUNT(*)

    return schema_dict_


def filter_dict_by_sl(schema_dict, sql):
    schema_dict_ = copy.deepcopy(schema_dict)
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    # tables
    for table in keys:
        if f'{table.lower()}' not in sql.lower():
            schema_dict_['tables'].pop(table, None)
    # columns
    keys = list(schema_dict_['tables'].keys())
    keys.sort(key=lambda x: - len(x))
    for table in keys:
        cls_keys = list(schema_dict_['tables'][table].keys())
        cls_keys.sort(key=lambda x: - len(x))
        tabel_dict = copy.deepcopy(schema_dict_['tables'][table])
        for cls in cls_keys:
            if cls.lower() not in sql.lower():
                schema_dict_['tables'][table].pop(cls, None)
        if len(schema_dict_['tables'][table].keys()) == 0:
            # schema_dict_['tables'][table] = tabel_dict  # for COUNT(*)
            for cls in tabel_dict.keys():
                if tabel_dict[cls][1] == True:
                    schema_dict_['tables'][table][cls] = tabel_dict[cls]

        if len(schema_dict_['tables'][table].keys()) == 0:
            schema_dict_['tables'][table][tabel_dict.keys()[0]] = tabel_dict[tabel_dict.keys()[0]]
            schema_dict_['tables'][table][tabel_dict.keys()[1]] = tabel_dict[tabel_dict.keys()[1]]
            # for COUNT(*)

    return schema_dict_


@func_set_timeout(5)
def execute_query_limit(db_path, query):
    error = ''
    result = None
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    cursor = conn.cursor()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result, error


def execute_query(db_path, query):
    try:
        result, error = execute_query_limit(db_path, query)
    except func_timeout.exceptions.FunctionTimedOut:
        error = "SQL execution timeout"
        print("*" * 30, error, query)
        result = None
    except Exception as e:
        error = str(e)
        print("*" * 30, error, query)
        result = None
    return result, error


def replace_syn(data1, data2):
    for i in range(len(data1)):
        if data1[i]['question'] == data2[i]['SpiderQuestion']:
            data1[i]['question'] = data2[i]['SpiderSynQuestion']
    return data1


def eval_all(args):
    dataset = args.dataset
    mode = args.mode
    data_tuples = parse_dataset(args.data_path, mode, dataset)
    batch_size = args.batch_size

    if dataset == 'spider_syn':
        data2 = read_json_file(os.path.join(args.data_path, 'spider', f'dev_syn.json'))
        data_tuples = replace_syn(data_tuples, data2)
        dataset = 'spider'
        args.tag += '_syn'

    if dataset == 'spider_DK':
        args.tag += '_DK'
        dataset = 'spider'

    if dataset == 'spider_real':
        args.tag += '_real'
        dataset = 'spider'

    if dataset == 'bird':
        kk = 5
    else:
        kk = 10
    kkkkk = 1 if dataset == 'bird' else 3

    if 'online' in args.tag:
        generator = LLM_Online()
    else:
        generator = LLM_Model(args.LLM_model)
    tag = args.tag


    # generate SQL
    if True:
        sql_results = []
        data_header = [["NLQ", "Predict", "GOLD", 'database']]
        prompts = []
        for index, row in enumerate(data_tuples):
            if 'spider' in dataset:
                row['SQL'] = row['query']
            if 'drspider' in dataset:
                row['SQL'] = row['query']

            question, db_id = row['question'], row['db_id']
            if dataset == 'spider':
                if mode == 'test':
                    db_path = os.path.join(args.data_path, dataset, 'test_database', db_id, f"{db_id}.sqlite")
                else:
                    db_path = os.path.join(args.data_path, dataset, 'database', db_id, f"{db_id}.sqlite")
            elif dataset == 'drspider':
                db_path = os.path.join(args.data_path, db_id, f"{db_id}.sqlite")
            elif dataset == 'bird':
                db_path = os.path.join(args.data_path, dataset, f'{mode}/{mode}_databases', db_id, f"{db_id}.sqlite")
            else:
                raise TypeError(f"Unexpect dataset: {dataset}.")

            schema_dict = get_schema_dict(db_path, kk=kk)
            database_schema, examples = get_schmea_str_and_examples(schema_dict)
            schema_dict_ = schema_dict

            if dataset == 'bird':
                prompt = [question, schema_dict,
                          f"\n\n/* Question hint */\n{row['evidence']}" if row['evidence'] != '' else '', schema_dict_]
            else:
                prompt = [question, schema_dict, '', schema_dict_]
            prompts.append([database_schema, str(examples), question, row['SQL'], db_id, prompt, db_path])

        n_samples = len(data_tuples)
        n_batches = (n_samples - 1) // batch_size + 1

        prompts_collection = []
        prompts_collection_db = []

        for i in range(n_batches):
            start = i * batch_size
            end = n_samples if i == n_batches - 1 else (i + 1) * batch_size
            batch_prompts = prompts[start: end]
            schema_dicts = []  # only keep the tables

            for j, v in enumerate(batch_prompts):
                batch_prompts[j][1] = get_example_str(batch_prompts[j][5][1], kkkkk)

            # text-to-sql
            final_prompts = [prompt_cw_temp_sft.format(ds=j[0], sr=j[1], qs=j[2], hint=j[5][2], sql='') for j in batch_prompts]
            response_strs = generator.generate_response(prompts=final_prompts)

            def contains_subquery(sql_query, tables):
                sql = sql_query.lower()
                select_num = 0
                join_num = 0
                tmp = sql
                while 'select' in tmp:
                    tmp = tmp[tmp.find('select') + 6:]
                    select_num += 1
                tmp = sql
                while 'join' in tmp:
                    tmp = tmp[tmp.find('select') + 6:]
                    join_num += 1
                table_num = len([key for key in tables if f"from {key.lower()}" in sql or f"join {key.lower()}" in sql])
                if table_num == 1:
                    hard = 1
                elif table_num == 2:
                    hard = 2
                else:
                    hard = 3
                return hard

            nc_idx = []
            continue_sqls = []
            # noisy correction

            for idx, v in enumerate(response_strs):
                pre_sql = parse_sql_from_string(response_strs[idx])
                ex_flg3 = True if execute_query(batch_prompts[idx][6], pre_sql)[1] == '' else False
                hard = contains_subquery(pre_sql, batch_prompts[idx][5][1]['tables'].keys())
                if ex_flg3 == False or hard > 2:
                    common_sql = 'SELECT '
                    continue_sqls.append(common_sql)
                    nc_idx.append(idx)

            # PSG
            if args.PSG:
                cl_prompts = []
                for j, idx in enumerate(nc_idx):
                    v = batch_prompts[idx]
                    ds = get_schmea_str_and_examples(v[5][1])[0]
                    sr = get_example_str(v[5][1], kkkkk)
                    common_sql = continue_sqls[j]
                    if args.eval_sft == 1:
                        cl_prompts.append(
                            prompt_cw_temp_sft.format(ds=ds, sr=sr, qs=v[2], hint=v[5][2], sql=common_sql))
                    else:
                        cl_prompts.append(prompt_cw_temp_sft.format(ds=ds, sr=sr, qs=v[2], hint=v[5][2], sql=common_sql))

                if len(nc_idx) > 0:
                    response_strs_ = generator.generate_response(prompts=cl_prompts)
                    print("%%%%%%%%%%%%%%%%%%", response_strs_[0])
                    for idx, v in enumerate(nc_idx):
                        if execute_query(batch_prompts[v][6], parse_sql_from_string(response_strs_[idx]))[
                            0] is not None:
                            response_strs[v] = response_strs_[idx]

            for j, response_str in enumerate(response_strs):
                database_schema = batch_prompts[j][0]
                question = batch_prompts[j][2]
                gt_sql = replace_multiple_spaces(batch_prompts[j][3])
                if gt_sql.endswith(";;"):
                    gt_sql = gt_sql[:-1]

                if not gt_sql.endswith(";"):
                    gt_sql += ";"

                db_id = batch_prompts[j][4]
                prompt = final_prompts[j]
                print(f"=={start + j + 1}/{len(data_tuples)}=={db_id}=={tag}==================")

                try:
                    if dataset == 'spider':
                        if mode == 'test':
                            db_path = os.path.join(args.data_path, dataset, 'test_database', db_id, f"{db_id}.sqlite")
                        else:
                            db_path = os.path.join(args.data_path, dataset, 'database', db_id, f"{db_id}.sqlite")
                    elif dataset == 'bird':
                        db_path = os.path.join(args.data_path, dataset, f'{mode}/{mode}_databases', db_id,
                                               f"{db_id}.sqlite")
                    else:
                        raise TypeError(f"Unexpect dataset: {dataset}.")

                    SQL_str = parse_sql_from_string(response_str)
                except Exception as e:
                    res = f'error: {str(e)}'
                    print(res, response_str)

                sql_results.append([question, SQL_str, gt_sql, db_id])

                if args.PSG:
                    if j in nc_idx:
                        prompt = prompt_cw_temp_sft.format(ds=batch_prompts[j][0], sr=batch_prompts[j][1], qs=batch_prompts[j][2], hint=batch_prompts[j][5][2], sql=truncate_sql_before_keywords(gt_sql, CLAUSE_KEYWORDS))
                        print(prompt)
                        print(f"Ground: {gt_sql}")
                        # input()

                        prompt_dict = {
                            "input": prompt,
                            "target": gt_sql
                        }

                        prompts_collection.append(prompt_dict)

                else:


                    print(prompt)
                    print(f"Ground: {gt_sql}")

                    prompt_dict1 = {
                        "input": prompt,
                        "db_id": db_id,
                        "target": gt_sql
                    }

                    prompt_dict2 = {
                        "input": prompt,
                        "target": gt_sql
                    }


                    prompts_collection_db.append(prompt_dict1)
                    prompts_collection.append(prompt_dict2)


        if args.PSG:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_{args.flags}_psg.json")

            with open(filename, mode='w',encoding='utf-8') as file:
                json.dump(prompts_collection, file, ensure_ascii=False, indent=4)
        else:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_{args.flags}.json")

            with open(filename, mode='w',encoding='utf-8') as file:
                json.dump(prompts_collection, file, ensure_ascii=False, indent=4)

        if prompts_collection_db:
            filename = os.path.join(args.output_path, f"{tag}_{dataset}_{mode}_db_id_{args.flags}.json")
            with open(filename, mode='w', encoding='utf-8') as file:
                json.dump(prompts_collection_db, file, ensure_ascii=False, indent=4)


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import pynvml

pynvml.nvmlInit()


def usegpu(need_gpu_count=1):
    nouse = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / meminfo.total
        if used < 0.3:
            nouse.append(index)

    if len(nouse) >= need_gpu_count:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse[:need_gpu_count]))
        # return nouse[:need_gpu_count]
        print(nouse[:need_gpu_count])
        return need_gpu_count
    elif len(nouse) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, nouse))
        return len(nouse)
    else:
        return 0


if __name__ == "__main__":
    from data_process import truncate_sql_before_keywords, CLAUSE_KEYWORDS
    parser = argparse.ArgumentParser(description='SQL')
    parser.add_argument("--dataset", default='bird', type=str)
    parser.add_argument("--data_path", default='/data/vda/dataset', type=str)
    parser.add_argument("--output_path", default='/data/vda/dataset', type=str)
    parser.add_argument("--mode", default='dev', type=str)
    parser.add_argument("--PSG", action='store_true', default=False)
    parser.add_argument("--tag", default='SQL-o1', type=str)
    parser.add_argument("--gpus", default=4, type=int)
    parser.add_argument("--eval_sft", default=1, type=int)
    parser.add_argument("--flags", default='0', type=str)
    # parser.add_argument("--LLM_model", default='meta-llama/Llama-3-8B-Instruct', type=str)
    parser.add_argument("--LLM_model", default='/data/vda/saves/llama3-8b', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    usegpu(need_gpu_count=args.gpus)
    print(args)
    eval_all(args)
```
