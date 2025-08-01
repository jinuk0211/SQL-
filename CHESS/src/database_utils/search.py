import logging
from typing import Dict
from langchain_chroma import Chroma

def query_vector_db(vector_db: Chroma, query: str, top_k: int) -> Dict[str, Dict[str, dict]]:
    """
    Queries the vector database for the most relevant documents based on the query.

    Args:
        vector_db (Chroma): The vector database to query.
        query (str): The query string to search for.
        top_k (int): The number of top results to return.

    Returns:
        Dict[str, Dict[str, dict]]: A dictionary containing table descriptions with their column details and scores.
    """
    table_description = {}
    
    try:
        relevant_docs_score = vector_db.similarity_search_with_score(query, k=top_k)
        logging.info(f"Query executed successfully: {query}")
    except Exception as e:
        logging.error(f"Error executing query: {query}, Error: {e}")
        raise e
    
    for doc, score in relevant_docs_score:
        metadata = doc.metadata
        table_name = metadata["table_name"]
        original_column_name = metadata["original_column_name"].strip()
        column_name = metadata["column_name"].strip()
        column_description = metadata["column_description"].strip()
        value_description = metadata["value_description"].strip()
        
        if table_name not in table_description:
            table_description[table_name] = {}
        
        if original_column_name not in table_description[table_name]:
            table_description[table_name][original_column_name] = {
                "column_name": column_name,
                "column_description": column_description,
                "value_description": value_description,
                "score": score
            }
    
    logging.info(f"Query results processed for query: {query}")
    return table_description

import pickle
from datasketch import MinHash, MinHashLSH
from pathlib import Path
import logging
from typing import Dict, Tuple, List

from database_utils.db_values.preprocess import _create_minhash

### Database value similarity ###

def _jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)

def load_db_lsh(db_directory_path: str) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
    """
    Loads the LSH and MinHashes from the preprocessed files in the specified directory.

    Args:
        db_directory_path (str): The path to the database directory.

    Returns:
        Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The LSH object and the dictionary of MinHashes.

    Raises:
        Exception: If there is an error loading the LSH or MinHashes.
    """
    db_id = Path(db_directory_path).name
    try:
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        with open(Path(db_directory_path) / "preprocessed" / f"{db_id}_minhashes.pkl", "rb") as file:
            minhashes = pickle.load(file)
        return lsh, minhashes
    except Exception as e:
        logging.error(f"Error loading LSH for {db_id}: {e}")
        raise e

def query_lsh(lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str, 
              signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """
    Queries the LSH for similar values to the given keyword and returns the top results.

    Args:
        lsh (MinHashLSH): The LSH object.
        minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
        keyword (str): The keyword to search for.
        signature_size (int, optional): The size of the MinHash signature.
        n_gram (int, optional): The n-gram size for the MinHash.
        top_n (int, optional): The number of top results to return.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
    """
    query_minhash = _create_minhash(signature_size, keyword, n_gram)
    results = lsh.query(query_minhash)
    similarities = [(result, _jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
    for result, similarity in similarities:
        table_name, column_name, value = minhashes[result][1:]
        if table_name not in similar_values_trimmed:
            similar_values_trimmed[table_name] = {}
        if column_name not in similar_values_trimmed[table_name]:
            similar_values_trimmed[table_name][column_name] = []
        similar_values_trimmed[table_name][column_name].append(value)

    return similar_values_trimmed