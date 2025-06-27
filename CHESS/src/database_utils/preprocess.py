import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from google.oauth2 import service_account
from google.cloud import aiplatform
import vertexai

from database_utils.db_catalog.csv_utils import load_tables_description

load_dotenv(override=True)

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
    aiplatform.init(
    project=GCP_PROJECT,
    location=GCP_REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
    )
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))


# EMBEDDING_FUNCTION = VertexAIEmbeddings(model_name="text-embedding-004")#OpenAIEmbeddings(model="text-embedding-3-large")
EMBEDDING_FUNCTION = OpenAIEmbeddings(model="text-embedding-3-large")


def make_db_context_vec_db(db_directory_path: str, **kwargs) -> None:
    """
    Creates a context vector database for the specified database directory.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs: Additional keyword arguments, including:
            - use_value_description (bool): Whether to include value descriptions (default is True).
    """
    db_id = Path(db_directory_path).name

    table_description = load_tables_description(db_directory_path, kwargs.get("use_value_description", True))
    docs = []
    
    for table_name, columns in table_description.items():
        for column_name, column_info in columns.items():
            metadata = {
                "table_name": table_name,
                "original_column_name": column_name,
                "column_name": column_info.get('column_name', ''),
                "column_description": column_info.get('column_description', ''),
                "value_description": column_info.get('value_description', '') if kwargs.get("use_value_description", True) else ""
            }
            for key in ['column_name', 'column_description', 'value_description']:
                if column_info.get(key, '').strip():
                    docs.append(Document(page_content=column_info[key], metadata=metadata))
    
    logging.info(f"Creating context vector database for {db_id}")
    vector_db_path = Path(db_directory_path) / "context_vector_db"

    if vector_db_path.exists():
        os.system(f"rm -r {vector_db_path}")

    vector_db_path.mkdir(exist_ok=True)

    Chroma.from_documents(docs, EMBEDDING_FUNCTION, persist_directory=str(vector_db_path))

    logging.info(f"Context vector database created at {vector_db_path}")
import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Any, Tuple

from database_utils.execution import execute_sql

def _get_unique_values(db_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Retrieves unique text values from the database excluding primary keys.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing unique values for each table and column.
    """
    table_names = [table[0] for table in execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';", fetch="all")]
    primary_keys = []

    for table_name in table_names:
        columns = execute_sql(db_path, f"PRAGMA table_info('{table_name}')", fetch="all")
        for column in columns:
            if column[5] > 0:  # Check if it's a primary key
                column_name = column[1]
                if column_name.lower() not in [c.lower() for c in primary_keys]:
                    primary_keys.append(column_name)
    
    unique_values: Dict[str, Dict[str, List[str]]] = {}
    for table_name in table_names:
        if table_name == "sqlite_sequence":
            continue
        logging.info(f"Processing {table_name}")
        columns = [col[1] for col in execute_sql(db_path, f"PRAGMA table_info('{table_name}')", fetch="all") if ("TEXT" in col[2] and col[1].lower() not in [c.lower() for c in primary_keys])]
        table_values: Dict[str, List[str]] = {}
        
        for column in columns:
            if any(keyword in column.lower() for keyword in ["_id", " id", "url", "email", "web", "time", "phone", "date", "address"]) or column.endswith("Id"):
                continue

            try:
                result = execute_sql(db_path, f"""
                    SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                    FROM (
                        SELECT DISTINCT `{column}` AS unique_values
                        FROM `{table_name}`
                        WHERE `{column}` IS NOT NULL
                    ) AS subquery
                """, fetch="one", timeout = 480)
            except:
                result = 0, 0

            sum_of_lengths, count_distinct = result
            if sum_of_lengths is None or count_distinct == 0:
                continue

            average_length = sum_of_lengths / count_distinct
            logging.info(f"Column: {column}, sum_of_lengths: {sum_of_lengths}, count_distinct: {count_distinct}, average_length: {average_length}")
            
            if ("name" in column.lower() and sum_of_lengths < 5000000) or (sum_of_lengths < 2000000 and average_length < 25) or count_distinct < 100:
                logging.info(f"Fetching distinct values for {column}")
                try:
                    values = [str(value[0]) for value in execute_sql(db_path, f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL", fetch="all", timeout = 480)]
                except:
                    values = []
                logging.info(f"Number of different values: {len(values)}")
                table_values[column] = values
        
        unique_values[table_name] = table_values

    return unique_values

def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
    """
    Creates a MinHash object for a given string.

    Args:
        signature_size (int): The size of the MinHash signature.
        string (str): The input string to create the MinHash for.
        n_gram (int): The n-gram size for the MinHash.

    Returns:
        MinHash: The MinHash object for the input string.
    """
    m = MinHash(num_perm=signature_size)
    for d in [string[i:i + n_gram] for i in range(len(string) - n_gram + 1)]:
        m.update(d.encode('utf8'))
    return m

def skip_column(column_name: str, column_values: List[str]) -> bool:
    """
    Determines whether to skip processing a column based on its values.

    Args:
        column_name (str): The name of the column.
        column_values (List[str]): The list of values in the column.

    Returns:
        bool: True if the column should be skipped, False otherwise.
    """
    if "name" in column_name.lower():
        return False
    sum_of_lengths = sum(len(value) for value in column_values)
    average_length = sum_of_lengths / len(column_values)
    return (sum_of_lengths > 50000) and (average_length > 20)

def make_lsh(unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float, verbose: bool = True) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Creates a MinHash LSH from unique values.

    Args:
        unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
        signature_size (int): The size of the MinHash signature.
        n_gram (int): The n-gram size for the MinHash.
        threshold (float): The threshold for the MinHash LSH.
        verbose (bool): Whether to display progress information.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
    minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
    try:
        total_unique_values = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        logging.info(f"Total unique values: {total_unique_values}")
        
        progress_bar = tqdm(total=total_unique_values, desc="Creating LSH") if verbose else None
        
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                if column_name.lower() == "doctype":
                    print("="*20)
                    print("Doctype found")
                    print("="*20)
                logging.info(f"Processing {table_name} - {column_name} - {len(column_values)}")
                
                for id, value in enumerate(column_values):
                    minhash = _create_minhash(signature_size, value, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{id}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh.insert(minhash_key, minhash)
                    
                    if verbose:
                        progress_bar.update(1)
        
        if verbose:
            progress_bar.close()
    except Exception as e:
        logging.error(f"Error creating LSH: {e}")
    
    return lsh, minhashes

def make_db_lsh(db_directory_path: str, **kwargs: Any) -> None:
    """
    Creates a MinHash LSH for the database and saves the results.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs (Any): Additional arguments for the LSH creation.
    """
    db_id = Path(db_directory_path).name
    preprocessed_path = Path(db_directory_path) / "preprocessed"
    preprocessed_path.mkdir(exist_ok=True)
    
    unique_values = _get_unique_values(str(Path(db_directory_path) / f"{db_id}.sqlite"))
    logging.info("Unique values obtained")
    
    with open(preprocessed_path / f"{db_id}_unique_values.pkl", "wb") as file:
        pickle.dump(unique_values, file)
    logging.info("Saved unique values")
    
    lsh, minhashes = make_lsh(unique_values, **kwargs)
    
    with open(preprocessed_path / f"{db_id}_lsh.pkl", "wb") as file:
        pickle.dump(lsh, file)
    with open(preprocessed_path / f"{db_id}_minhashes.pkl", "wb") as file:
        pickle.dump(minhashes, file)