"""
Aparecium Training and Demonstration Script

This script demonstrates how to use Aparecium's Vectorizer and Seq2SeqReverser classes
in conjunction with a ApareciumDB instance. It covers:

1. Retrieving or generating sample training data (sentences + embeddings) within a given
   block range.
2. Training a seq2seq model to reverse embeddings back to text using teacher forcing.
3. Testing the model on sample sentences with different decoding strategies (greedy,
   beam search, sampling).
4. Periodically saving and loading the model checkpoints to/from disk.
5. Storing all data (sentences & embeddings) in an SQLite database for persistence.

Run this script directly (e.g. `python your_script.py`) to see how the
workflow operates in practice.
"""

from typing import Tuple, List
import os
import random
import torch  # type: ignore

from aparecium import Vectorizer, Seq2SeqReverser  # type: ignore
from aparecium.db_utils import ApareciumDB  # type: ignore


def prepare_train_data(
    block_start: int,
    block_end: int,
    vectorizer: Vectorizer,
    db: ApareciumDB,
) -> Tuple[List[str], List[List[List[float]]]]:
    """
    Prepare training data (sentences and embeddings) for a given block range.

    This function first checks if data for the specified block range exists
    in the SQLite database. If so, it retrieves the existing sentences and
    embeddings. Otherwise, it generates a set of sample sentences, encodes
    them into embeddings via the provided vectorizer, and stores them in
    the database.

    Args:
        block_start (int):
            The starting block number for the data range.
        block_end (int):
            The ending block number for the data range.
        vectorizer (Vectorizer):
            The vectorizer used to encode text into embeddings.
        db (ApareciumDB):
            The database object used for checking/storing/retrieving data.

    Returns:
        Tuple[List[str], List[List[List[float]]]]:
            A tuple containing two elements:
            1) A list of sentences (strings).
            2) A corresponding list of embedding matrices, where each matrix is
               a 2D list of floats (shape: [sequence_length, embedding_dimension]).
    """
    if db.check_batch_exists(block_start, block_end):
        train_sentences, train_matrices = db.retrieve_batch(block_start, block_end)
        print(
            f"Loaded {len(train_sentences)} sentences and matrices from database for blocks {block_start}-{block_end}"
        )
    else:
        print("No existing data found. Creating sample data for demonstration...")
        sample_sentences = [
            "I really enjoy using this blockchain platform for sharing thoughts.",
            "Just bought some new tokens and feeling optimistic about the market!",
            "This new DeFi protocol looks promising, what do you all think?",
            "Crypto prices are looking volatile today, hodl strong everyone.",
            "I'm excited about the future of decentralized social media!",
            "Web3 technologies are transforming how we interact online.",
            "Learning about smart contracts has been a fascinating journey.",
            "Just published my first NFT collection, check it out!",
            "The community support in this ecosystem is amazing.",
            "Thinking about the long-term implications of digital ownership.",
        ]

        train_sentences = []
        train_matrices = []
        block_numbers = []
        transaction_ids = []

        total_samples = min(
            50, (block_end - block_start + 1) * 5
        )  # Approximately 5 per block

        for i in range(total_samples):
            if random.random() > 0.5 and sample_sentences:
                post_content = random.choice(sample_sentences)
            else:
                post_content = f"Sample post #{i} about blockchain technology and its applications."

            train_sentences.append(post_content)
            train_matrices.append(vectorizer.encode(post_content))
            block_numbers.append(random.randint(block_start, block_end))
            transaction_ids.append(f"sample_tx_{i}")

        db.store_batch(
            block_start=block_start,
            block_end=block_end,
            sentences=train_sentences,
            matrices=train_matrices,
            block_numbers=block_numbers,
            transaction_ids=transaction_ids,
        )
        print(
            f"Stored {len(train_sentences)} sample sentences and matrices in database for demonstration"
        )

    return train_sentences, train_matrices


def test_sample(vectorizer: Vectorizer, reverser: Seq2SeqReverser) -> None:
    """
    Test the seq2seq Reverser with sample sentences using different decoding strategies.

    This function encodes a list of predefined test sentences into embeddings
    using the provided Vectorizer, then uses the Seq2SeqReverser to generate
    text for each embedding with three decoding methods:
      1. Greedy decoding
      2. Beam search decoding
      3. Sampling-based decoding (with top-k, top-p, and temperature)

    The results are printed for comparison.

    Args:
        vectorizer (Vectorizer):
            The vectorizer responsible for encoding text into numeric embeddings.
        reverser (Seq2SeqReverser):
            The seq2seq model that decodes embeddings back into text.

    Returns:
        None
    """
    test_sentences = [
        "This is a test sentence about blockchain technology!",
        "How might decentralized systems change the future of finance?",
        "Web3 innovations are creating new opportunities for creators.",
    ]

    for test_s in test_sentences:
        test_matrix = vectorizer.encode(test_s)

        generated_greedy = reverser.generate_text(
            test_matrix,
            max_length=50,
            num_beams=1,  # Greedy if num_beams=1
            do_sample=False,  # No sampling => pure greedy
        )

        generated_beam = reverser.generate_text(
            test_matrix,
            max_length=50,
            num_beams=3,  # Use 3 beams
            do_sample=False,
        )

        generated_sample = reverser.generate_text(
            test_matrix,
            max_length=50,
            num_beams=1,  # Only 1 beam, but do sampling
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0,
        )

        print("---")
        print(f"Original:   {test_s}")
        print(f"Greedy:     {generated_greedy}")
        print(f"BeamSearch: {generated_beam}")
        print(f"Sampling:   {generated_sample}")
        print("---")


def main():
    """
    Main function demonstrating an end-to-end usage of the Aparecium package.

    Steps performed:
    1. Define parameters and block ranges.
    2. Initialize the SQLite database, vectorizer, and seq2seq reverser.
    3. Attempt to load an existing trained model from disk; otherwise train from scratch.
    4. For a specified number of epochs:
       a) Loop through the block ranges in increments (block_size).
       b) Retrieve or generate training data via `prepare_train_data`.
       c) Train the seq2seq reverser in batched mode on these samples.
       d) Periodically test the model's performance with `test_sample`.
       e) Save model checkpoints after processing each block.
    5. Print an average epoch loss, then close the database connection.

    Note:
        This script is for demonstration. For a production scenario, you may want
        to adjust hyperparameters (e.g., EPOCHS, BATCH_SIZE) and handle your own
        dataset rather than the generated samples.

    Returns:
        None
    """
    # 1) Parameters
    block_start = 1
    block_end = 100
    block_size = 50

    # 2) Initialize device, database, vectorizer, and reverser
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    db = ApareciumDB(db_path="data/example.db")
    print("Initialized database at data/example.db")

    vectorizer = Vectorizer(
        model_name="sentence-transformers/all-mpnet-base-v2",
        device=device,
    )

    reverser = Seq2SeqReverser(
        model_name="sentence-transformers/all-mpnet-base-v2",
        d_model=768,
        num_decoder_layers=2,
        nhead=8,
        dim_feedforward=2048,
        lr=1e-4,
        device=device,
    )

    # 3) Attempt to load pre-trained model checkpoint, if available
    model_path = "models/seq2seqreverser/example"
    try:
        reverser.load_model(model_path, device)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
        print("Will train from scratch")

    # 4) Training loop (demonstration)
    EPOCHS = 5
    BATCH_SIZE = 8

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_samples = 0

        # Process data in sub-ranges
        for start in range(block_start, block_end, block_size):
            current_end = min(start + block_size - 1, block_end)
            print(f"Processing blocks {start}-{current_end}")

            # 4A) Retrieve or generate training data
            train_sentences, train_matrices = prepare_train_data(
                start, current_end, vectorizer, db
            )
            block_num_samples = len(train_sentences)

            if block_num_samples == 0:
                print(f"No samples found for blocks {start}-{current_end}, skipping")
                continue

            # 4B) Train in mini-batches
            for i in range(0, block_num_samples, BATCH_SIZE):
                batch_sents = train_sentences[i : i + BATCH_SIZE]
                batch_mats = train_matrices[i : i + BATCH_SIZE]

                # One batched training step
                loss = reverser.train_step_batch(
                    source_rep_batch=batch_mats,
                    target_text_batch=batch_sents,
                    max_source_length=256,
                    max_target_length=256,
                )
                # Accumulate total loss
                effective_batch_size = len(batch_sents)
                total_loss += loss * effective_batch_size
                total_samples += effective_batch_size

                # Print progress
                if i % (BATCH_SIZE * 10) == 0 or i + BATCH_SIZE >= block_num_samples:
                    print(
                        f"Epoch {epoch+1}/{EPOCHS}, Block {start}-{current_end}, "
                        f"Batch {i//BATCH_SIZE + 1}/{(block_num_samples + BATCH_SIZE - 1)//BATCH_SIZE}, "
                        f"Loss: {loss:.4f}"
                    )

            # Test model on a small set of sample sentences
            test_sample(vectorizer, reverser)

            # Save a checkpoint after each block range
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            reverser.save_model(model_path)
            print(f"Model saved to {model_path}")

        # Compute average loss for this epoch
        avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

    # Close the database connection
    db.close()


if __name__ == "__main__":
    main()
