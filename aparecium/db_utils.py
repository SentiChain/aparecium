"""
Database Utilities for Aparecium

This module provides functionality for storing and retrieving sentence data
and associated embeddings in an SQLite database. It includes:

1. **NumpyArrayAdapter**:
    - A helper class to serialize/deserialize NumPy arrays to/from SQLite BLOB fields.

2. **TorchTensorAdapter**:
    - A helper class to serialize/deserialize PyTorch tensors to/from SQLite BLOB fields.

3. **ApareciumDB**:
    - A high-level database utility class that handles table creation, 
      insertion, and retrieval of sentences and their corresponding 
      matrices (NumPy arrays or PyTorch tensors).

Typical usage includes:
- Initializing a ApareciumDB instance with a SQLite file path.
- Storing batches of sentences and their embeddings.
- Retrieving batches for a given block range.
- Checking existence or size of a particular batch of data.

Example:

    from db_utils import ApareciumDB, NumpyArrayAdapter, TorchTensorAdapter
    import sqlite3
    import numpy as np
    import torch

    # Optional: register adapters with SQLite if needed
    sqlite3.register_adapter(np.ndarray, NumpyArrayAdapter.adapt)
    sqlite3.register_converter("BLOB", NumpyArrayAdapter.convert)
    sqlite3.register_adapter(torch.Tensor, TorchTensorAdapter.adapt)
    sqlite3.register_converter("BLOB", TorchTensorAdapter.convert)

    # Initialize the database
    db = ApareciumDB("data/my_database.db")

    # Store a batch of sentences and matrices
    sentences = ["Hello world", "Another example"]
    arrays = [np.random.rand(5, 5), torch.rand(3, 3)]
    db.store_batch(
        block_start=0,
        block_end=1,
        sentences=sentences,
        matrices=arrays,
    )

    # Retrieve the stored batch
    retrieved_sentences, retrieved_matrices = db.retrieve_batch(block_start=0, block_end=1)

    # Close the database connection
    db.close()
"""

from typing import List, Tuple, Optional, Any
import os
import io
import sqlite3
import torch  # type: ignore
import numpy as np  # type: ignore

from .logger import logger  # type: ignore
from .exceptions import DatabaseError, ConfigurationError  # type: ignore


class NumpyArrayAdapter:
    """
    Adapter for converting between NumPy arrays and SQLite BLOB data.

    This class provides two static methods:
      1. `adapt(arr)`: Converts a NumPy array to a binary blob for storage.
      2. `convert(blob)`: Converts a binary blob back into a NumPy array.

    When registering adapters with SQLite, you can do something like:

        sqlite3.register_adapter(np.ndarray, NumpyArrayAdapter.adapt)
        sqlite3.register_converter("BLOB", NumpyArrayAdapter.convert)
    """

    @staticmethod
    def adapt(arr):
        """
        Convert a NumPy array into binary data suitable for SQLite storage.

        Args:
            arr (np.ndarray):
                The NumPy array to be converted.

        Returns:
            sqlite3.Binary:
                A binary object containing the serialized NumPy array.
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert(blob):
        """
        Convert a binary blob (serialized NumPy array) back into a NumPy array.

        Args:
            blob (bytes):
                A bytes object that was previously serialized by `np.save()`.

        Returns:
            np.ndarray:
                The deserialized NumPy array.
        """
        out = io.BytesIO(blob)
        out.seek(0)
        return np.load(out)


class TorchTensorAdapter:
    """
    Adapter for converting between PyTorch tensors and SQLite BLOB data.

    This class provides two static methods:
      1. `adapt(tensor)`: Converts a PyTorch tensor to a binary blob for storage.
      2. `convert(blob)`: Converts a binary blob back into a PyTorch tensor.

    When registering adapters with SQLite, you can do something like:

        sqlite3.register_adapter(torch.Tensor, TorchTensorAdapter.adapt)
        sqlite3.register_converter("BLOB", TorchTensorAdapter.convert)
    """

    @staticmethod
    def adapt(tensor):
        """
        Convert a PyTorch tensor into binary data suitable for SQLite storage.

        Args:
            tensor (torch.Tensor):
                The PyTorch tensor to be converted.

        Returns:
            sqlite3.Binary:
                A binary object containing the serialized PyTorch tensor.
        """
        out = io.BytesIO()
        torch.save(tensor, out)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert(blob):
        """
        Convert a binary blob (serialized PyTorch tensor) back into a torch.Tensor.

        Args:
            blob (bytes):
                A bytes object that was previously serialized by `torch.save()`.

        Returns:
            torch.Tensor:
                The deserialized PyTorch tensor.
        """
        out = io.BytesIO(blob)
        out.seek(0)
        return torch.load(out, weights_only=True)


class ApareciumDB:
    """
    Database utility for storing and retrieving sentences and associated matrices.

    This class manages an SQLite database that stores:
      - Sentence text along with metadata (block range, transaction IDs, etc.).
      - Matrices (either NumPy arrays or PyTorch tensors) associated with each sentence.

    The database schema includes two tables:
      1. `sentences`:
         - `id` (PRIMARY KEY)
         - `block_start`, `block_end` (block range info)
         - `sentence` (text content)
         - `block_number` (optional, user-provided block index)
         - `transaction_id` (optional, user-provided transaction string)

      2. `matrices`:
         - `id` (PRIMARY KEY)
         - `sentence_id` (FOREIGN KEY referencing `sentences.id`)
         - `matrix` (BLOB field for serialized NumPy array or PyTorch tensor)

    Attributes:
        db_path (str): The path to the SQLite database file.
        conn (sqlite3.Connection): The active connection to the SQLite database.
    """

    def __init__(self, db_path="data/aparecium.db"):
        """
        Initialize a ApareciumDB instance.

        Ensures that the specified directory for `db_path` exists,
        establishes a connection to the database, and creates the
        necessary tables (if they do not already exist).

        Args:
            db_path (str, optional):
                File path of the SQLite database.
                Defaults to "data/aparecium.db".

        Raises:
            ConfigurationError: If the database path is invalid or directory creation fails.
            DatabaseError: If database connection or table creation fails.
        """
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            logger.debug(f"Initializing database at path: {db_path}")

            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            logger.info(f"Successfully connected to database at {db_path}")

            # Create tables if they don't exist
            self._create_tables()
        except OSError as e:
            logger.error(f"Failed to create database directory: {str(e)}")
            raise ConfigurationError(
                f"Invalid database path or directory creation failed: {str(e)}"
            )
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")

    def _create_tables(self):
        """
        Create the required tables for storing sentences and matrices, if not present.

        This method also creates an index on `(block_start, block_end)` for
        potentially faster queries on that range.

        Raises:
            DatabaseError: If table creation fails.
        """
        try:
            with self.conn:
                # Create tables for blocks and transactions
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sentences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        block_start INTEGER,
                        block_end INTEGER,
                        sentence TEXT,
                        block_number INTEGER,
                        transaction_id TEXT
                    )
                """
                )

                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS matrices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sentence_id INTEGER,
                        matrix BLOB,
                        FOREIGN KEY (sentence_id) REFERENCES sentences (id)
                    )
                """
                )

                # Create indices for faster retrieval
                self.conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_block_range 
                    ON sentences (block_start, block_end)
                """
                )
                logger.debug(
                    "Successfully created/verified database tables and indices"
                )
        except sqlite3.Error as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise DatabaseError(f"Failed to create database tables: {str(e)}")

    def store_batch(
        self,
        block_start: int,
        block_end: int,
        sentences: List[str],
        matrices: List[Any],
        block_numbers: Optional[List[int]] = None,
        transaction_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Store a batch of sentences and their corresponding matrices in the database.

        Each sentence is inserted into the `sentences` table, and each matrix
        (NumPy array or PyTorch tensor) is stored in the `matrices` table with
        a matching `sentence_id`.

        Args:
            block_start (int):
                The starting block number for this batch.
            block_end (int):
                The ending block number for this batch.
            sentences (List[str]):
                A list of sentence strings to store.
            matrices (List[Any]):
                A list of matrix objects (e.g., numpy.ndarray or torch.Tensor).
                Must have the same length as `sentences`.
            block_numbers (Optional[List[int]]):
                Optional list of block numbers for each sentence. Defaults to None.
            transaction_ids (Optional[List[str]]):
                Optional list of transaction IDs for each sentence. Defaults to None.

        Raises:
            ValueError: If lengths of `sentences` and `matrices` differ.
            DatabaseError: If database operations fail.
        """
        if len(sentences) != len(matrices):
            logger.error(
                f"Mismatch in batch sizes: {len(sentences)} sentences vs {len(matrices)} matrices"
            )
            raise ValueError("Lengths of sentences and matrices must match")

        if block_numbers is None:
            block_numbers = [None] * len(sentences)

        if transaction_ids is None:
            transaction_ids = [None] * len(sentences)

        try:
            with self.conn:
                # First, store all sentences and get their IDs
                sentence_ids = []
                for i, sentence in enumerate(sentences):
                    cursor = self.conn.execute(
                        """
                        INSERT INTO sentences (block_start, block_end, sentence, block_number, transaction_id)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            block_start,
                            block_end,
                            sentence,
                            block_numbers[i],
                            transaction_ids[i],
                        ),
                    )
                    sentence_ids.append(cursor.lastrowid)

                # Now store all matrices with their corresponding sentence IDs
                for sentence_id, matrix in zip(sentence_ids, matrices):
                    # Determine type and convert accordingly
                    if isinstance(matrix, torch.Tensor):
                        matrix_blob = TorchTensorAdapter.adapt(matrix)
                    else:
                        matrix_blob = NumpyArrayAdapter.adapt(matrix)

                    self.conn.execute(
                        """
                        INSERT INTO matrices (sentence_id, matrix)
                        VALUES (?, ?)
                    """,
                        (sentence_id, matrix_blob),
                    )
                logger.debug(
                    f"Successfully stored batch of {len(sentences)} sentences and matrices"
                )
        except sqlite3.Error as e:
            logger.error(f"Failed to store batch in database: {str(e)}")
            raise DatabaseError(f"Failed to store batch in database: {str(e)}")

    def retrieve_batch(
        self, block_start: int, block_end: int
    ) -> Tuple[List[str], List[Any]]:
        """
        Retrieve sentences and matrices for a specific block range.

        Queries the `sentences` table for records matching the given `(block_start, block_end)`
        and then retrieves the associated matrices from the `matrices` table.

        Args:
            block_start (int):
                The starting block number.
            block_end (int):
                The ending block number.

        Returns:
            Tuple[List[str], List[Any]]:
                A tuple `(sentences, matrices)` where:
                  - `sentences` is a list of sentence strings.
                  - `matrices` is a list of deserialized NumPy arrays or PyTorch tensors
                    in the same order as `sentences`. If no matrix is found for a
                    particular sentence, `None` is placed in its position.

        Raises:
            DatabaseError: If database operations fail.
        """
        try:
            with self.conn:
                # First get all sentences in the block range
                cursor = self.conn.execute(
                    """
                    SELECT id, sentence
                    FROM sentences
                    WHERE block_start = ? AND block_end = ?
                """,
                    (block_start, block_end),
                )

                results = cursor.fetchall()
                if not results:
                    logger.debug(
                        f"No data found for block range {block_start}-{block_end}"
                    )
                    return [], []

                sentence_ids = [row[0] for row in results]
                sentences = [row[1] for row in results]

                # Now get all matrices for these sentence IDs
                # Build a parameterized query with the right number of placeholders
                placeholders = ",".join(["?"] * len(sentence_ids))
                query = f"""
                    SELECT sentence_id, matrix
                    FROM matrices
                    WHERE sentence_id IN ({placeholders})
                    ORDER BY sentence_id
                """

                cursor = self.conn.execute(query, sentence_ids)
                matrix_results = cursor.fetchall()

                # Create a mapping from sentence_id to matrix for proper ordering
                matrix_map = {row[0]: row[1] for row in matrix_results}

                # Order matrices to match the sentences
                matrices = []
                for sid in sentence_ids:
                    if sid in matrix_map:
                        # Try first as PyTorch tensor, fall back to NumPy
                        try:
                            matrix = TorchTensorAdapter.convert(matrix_map[sid])
                        except:
                            matrix = NumpyArrayAdapter.convert(matrix_map[sid])
                        matrices.append(matrix)
                    else:
                        matrices.append(None)  # Handle missing matrices

                logger.debug(
                    f"Successfully retrieved batch of {len(sentences)} sentences and matrices"
                )
                return sentences, matrices
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve batch from database: {str(e)}")
            raise DatabaseError(f"Failed to retrieve batch from database: {str(e)}")

    def check_batch_exists(self, block_start: int, block_end: int) -> bool:
        """
        Check whether a batch exists for the given block range.

        Args:
            block_start (int):
                The starting block number.
            block_end (int):
                The ending block number.

        Returns:
            bool:
                True if at least one sentence is stored with the specified block range,
                False otherwise.

        Raises:
            DatabaseError: If database operations fail.
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) 
                    FROM sentences
                    WHERE block_start = ? AND block_end = ?
                """,
                    (block_start, block_end),
                )

                count = cursor.fetchone()[0]
                exists = count > 0
                logger.debug(
                    f"Batch existence check for range {block_start}-{block_end}: {exists}"
                )
                return exists
        except sqlite3.Error as e:
            logger.error(f"Failed to check batch existence: {str(e)}")
            raise DatabaseError(f"Failed to check batch existence: {str(e)}")

    def get_batch_size(self, block_start: int, block_end: int) -> int:
        """
        Get the number of sentences in a specific batch.

        Args:
            block_start (int):
                The starting block number.
            block_end (int):
                The ending block number.

        Returns:
            int:
                The count of sentences stored for the specified block range.

        Raises:
            DatabaseError: If database operations fail.
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) 
                    FROM sentences
                    WHERE block_start = ? AND block_end = ?
                """,
                    (block_start, block_end),
                )

                count = cursor.fetchone()[0]
                logger.debug(f"Batch size for range {block_start}-{block_end}: {count}")
                return count
        except sqlite3.Error as e:
            logger.error(f"Failed to get batch size: {str(e)}")
            raise DatabaseError(f"Failed to get batch size: {str(e)}")

    def close(self):
        """
        Close the database connection.

        This method can be called explicitly or will be invoked automatically
        when the object is garbage-collected via `__del__`.

        Raises:
            DatabaseError: If closing the connection fails.
        """
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info("Database connection closed successfully")
            except sqlite3.Error as e:
                logger.error(f"Failed to close database connection: {str(e)}")
                raise DatabaseError(f"Failed to close database connection: {str(e)}")

    def __del__(self):
        """
        Destructor to ensure the database connection is closed.

        Automatically called when the `ApareciumDB` instance is garbage-collected.
        """
        self.close()
