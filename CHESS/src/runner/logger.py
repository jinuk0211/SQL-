import logging
import json
from threading import Lock
from pathlib import Path
from typing import Any, List, Dict, Union

from runner.run_manager import Task

class Logger:
    _instance = None
    _lock = Lock()

    def __new__(cls, db_id: str = None, question_id: str = None, result_directory: str = None):
        """
        Ensures a singleton instance of Logger.

        Args:
            db_id (str, optional): The database ID.
            question_id (str, optional): The question ID.
            result_directory (str, optional): The directory to store results.

        Returns:
            Logger: The singleton instance of the class.

        Raises:
            ValueError: If the Logger instance has not been initialized.
        """
        with cls._lock:
            if (db_id is not None) and (question_id is not None):
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance._init(db_id, question_id, result_directory)
            else:
                if cls._instance is None:
                    raise ValueError("Logger instance has not been initialized.")
            return cls._instance

    def _init(self, db_id: str, question_id: str, result_directory: str):
        """
        Initializes the Logger instance with the provided parameters.

        Args:
            db_id (str): The database ID.
            question_id (str): The question ID.
            result_directory (str): The directory to store results.
        """
        self.db_id = db_id
        self.question_id = question_id
        self.result_directory = Path(result_directory)
        self.log_file_lock = Lock()

    def _set_log_level(self, log_level: str):
        """
        Sets the logging level.

        Args:
            log_level (str): The logging level to set.

        Raises:
            ValueError: If the log level is invalid.
        """
        log_level_attr = getattr(logging, log_level.upper(), None)
        if log_level_attr is None:
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(level=log_level_attr, format='%(levelname)s: %(message)s')

    def log(self, message: str, log_level: str = "info", task: Task = None):
        """
        Logs a message at the specified log level.

        Args:
            message (str): The message to log.
            log_level (str): The log level to use.

        Raises:
            ValueError: If the log level is invalid.
        """
        log_method = getattr(logging, log_level, None)
        if log_method is None:
            raise ValueError(f"Invalid log level: {log_level}")
        if task is not None:
            log_method(f"({task.db_id}, {task.question_id}) {message}")
        else:
            log_method(message)

    def log_conversation(self, conversations: List[Dict[str, Any]]):
        """
        Logs conversations to a file.

        Args:
            conversations (List[Dict[str, Any]]): The conversations to log.
        """
        with self.log_file_lock:
            log_file_path = self.result_directory / "logs" / f"{self.question_id}_{self.db_id}.log"
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with log_file_path.open("a") as file:
                for conversation in conversations:
                    text = conversation["text"]
                    file.write(f"############################## {conversation['from']} at step {conversation['step']} ##############################\n\n")
                    if isinstance(text, str):
                        file.write(text)
                    elif isinstance(text, (list, dict)):
                        formatted_text = json.dumps(text, indent=4)
                        file.write(formatted_text)
                    elif isinstance(text, bool):
                        file.write(str(text))
                    file.write("\n\n")

    def dump_history_to_file(self, execution_history: List[Dict[str, Any]]):
        """
        Dumps the execution history to a JSON file.

        Args:
            execution_history (List[Dict[str, Any]]): The execution history to dump.
        """
        file_path = self.result_directory / f"{self.question_id}_{self.db_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as file:
            json.dump(execution_history, file, indent=4)

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

@dataclass
class Statistics:
    corrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    incorrects: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    errors: Dict[str, List[Union[Tuple[str, str], Tuple[str, str, str]]]] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]:
        """
        Converts the statistics data to a dictionary format.

        Returns:
            Dict[str, Dict[str, Union[Dict[str, int], List[Tuple[str, str]]]]]: The statistics data as a dictionary.
        """
        return {
            "counts": {
                key: {
                    "correct": len(self.corrects.get(key, [])),
                    "incorrect": len(self.incorrects.get(key, [])),
                    "error": len(self.errors.get(key, [])),
                    "total": self.total.get(key, 0)
                }
                for key in self.total
            },
            "ids": {
                key: {
                    "correct": sorted(self.corrects.get(key, [])),
                    "incorrect": sorted(self.incorrects.get(key, [])),
                    "error": sorted(self.errors.get(key, []))
                }
                for key in self.total
            }
        }

class StatisticsManager:
    def __init__(self, result_directory: str):
        """
        Initializes the StatisticsManager.

        Args:
            result_directory (str): The directory to store results.
        """
        self.result_directory = Path(result_directory)
        self.statistics = Statistics()

        # Ensure the statistics file exists
        self.statistics_file_path = self.result_directory / "-statistics.json"
        if not self.statistics_file_path.exists():
            self.statistics_file_path.touch()
            self.dump_statistics_to_file()

    def update_stats(self, db_id: str, question_id: str, validation_for: str, result: Dict[str, Any]):
        """
        Updates the statistics based on the validation result.

        Args:
            db_id (str): The database ID.
            question_id (str): The question ID.
            validation_for (str): The validation context.
            result (Dict[str, Any]): The validation result.
        """
        exec_res = result["exec_res"]
        exec_err = result["exec_err"]

        self.statistics.total[validation_for] = self.statistics.total.get(validation_for, 0) + 1

        if exec_res == 1:
            if validation_for not in self.statistics.corrects:
                self.statistics.corrects[validation_for] = []
            self.statistics.corrects[validation_for].append((db_id, question_id))
        else:
            if exec_err == "incorrect answer":
                if validation_for not in self.statistics.incorrects:
                    self.statistics.incorrects[validation_for] = []
                self.statistics.incorrects[validation_for].append((db_id, question_id))
            else:
                if validation_for not in self.statistics.errors:
                    self.statistics.errors[validation_for] = []
                self.statistics.errors[validation_for].append((db_id, question_id, exec_err))

    def dump_statistics_to_file(self):
        """
        Dumps the current statistics to a JSON file.
        """
        with self.statistics_file_path.open('w') as f:
            json.dump(self.statistics.to_dict(), f, indent=4)