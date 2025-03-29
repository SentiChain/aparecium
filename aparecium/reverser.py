"""
Seq2Seq Reverser Module

This module provides functionality for converting numeric representations back to text
using a Transformer-based sequence-to-sequence architecture. It includes a decoder model 
that can be trained with teacher forcing and used for inference to generate text from 
embedded representations.

The core classes include:
- TransformerSeq2SeqModel: The neural network model for decoding
- Seq2SeqReverser: Main interface for training and text generation

Example usage:
    reverser = Seq2SeqReverser()
    # Train with embedded representations and corresponding text
    reverser.train_step(source_embeddings, target_text)
    # Generate text from embeddings
    generated_text = reverser.generate_text(source_embeddings)
"""

from typing import Optional, List
import os
import logging
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch._dynamo  # type: ignore

logger = logging.getLogger(__name__)


def generate_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal (autoregressive) mask of shape (sz, sz).

    This mask ensures each position can only attend to previous positions,
    which is necessary for autoregressive decoding. The resulting mask is a boolean
    tensor where True values indicate positions that should be masked out (cannot
    attend to).

    Args:
        sz: The size of the square mask
        device: The torch device on which to create the mask

    Returns:
        A boolean tensor of shape (sz, sz) where True values indicate positions
        that should be masked out (i.e., future tokens that cannot be attended to)
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask


class TransformerSeq2SeqModel(nn.Module):
    """
    A Transformer decoder that consumes 'memory' from an encoder
    and autoregressively produces output tokens.

    This model implements a standard Transformer decoder architecture with
    token embeddings, positional embeddings, and a transformer decoder stack.
    It takes encoded representations (memory) as input and generates a sequence
    of output tokens.

    Attributes:
        token_embedding: Embedding layer for input tokens
        pos_embedding: Positional embedding layer
        transformer_decoder: Stack of transformer decoder layers
        fc_out: Linear layer projecting to vocabulary size
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
    ):
        """
        Initialize the TransformerSeq2SeqModel.

        Args:
            vocab_size: Size of the vocabulary (output dimension)
            d_model: Dimensionality of the model's hidden states
            num_decoder_layers: Number of stacked transformer decoder layers
            nhead: Number of attention heads in the transformer
            dim_feedforward: Dimensionality of the transformer's feed-forward networks
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer decoder model.

        Args:
            encoder_outputs: Output tensor from an encoder (memory)
                Shape: (src_seq_len, batch_size, d_model)
            tgt_input_ids: Target input token IDs
                Shape: (tgt_seq_len, batch_size)
            tgt_mask: Mask to prevent attending to future positions
                Shape: (tgt_seq_len, tgt_seq_len)

        Returns:
            Logits for the next token prediction
                Shape: (tgt_seq_len, batch_size, vocab_size)
        """
        tgt_seq_len, batch_size = tgt_input_ids.size()

        token_emb = self.token_embedding(tgt_input_ids)
        positions = torch.arange(tgt_seq_len, device=tgt_input_ids.device).unsqueeze(1)
        pos_emb = self.pos_embedding(positions).squeeze(1)
        token_emb = token_emb + pos_emb.unsqueeze(1)

        hidden_states = self.transformer_decoder(
            tgt=token_emb,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
        )
        logits = self.fc_out(hidden_states)
        return logits


class Seq2SeqReverser:
    """
    A seq2seq model that takes a numeric "source" representation
    (list of lists of floats) and produces text.

    This class provides the main interface for the reverser functionality,
    handling training, inference, saving, and loading of models. It can be
    trained with teacher forcing by providing numeric encoder outputs and
    target text pairs.

    Attributes:
        device: The torch device (CPU or GPU) to use
        tokenizer: A pretrained tokenizer for converting between text and token IDs
        decoder: The TransformerSeq2SeqModel instance
        criterion: Loss function (typically CrossEntropyLoss)
        optimizer: Optimizer for training
        config: Dictionary storing model configuration parameters
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        d_model: int = 768,
        num_decoder_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        """
        Initialize the Seq2SeqReverser model.

        Args:
            model_name: The name or path of the pre-trained model to use for the tokenizer
            d_model: Dimensionality of the model's hidden states
            num_decoder_layers: Number of stacked transformer decoder layers
            nhead: Number of attention heads in the transformer
            dim_feedforward: Dimensionality of the transformer's feed-forward networks
            lr: Learning rate for the optimizer
            device: The device to use ('cuda', 'cpu', or None to auto-select)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Use the same tokenizer that was used for the embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create the decoder
        vocab_size = len(self.tokenizer)
        self.decoder = TransformerSeq2SeqModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = optim.AdamW(self.decoder.parameters(), lr=lr)

        self.config = {
            "model_name": model_name,
            "d_model": d_model,
            "num_decoder_layers": num_decoder_layers,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "lr": lr,
        }

    def train_step(self, source_rep: List[List[float]], target_text: str) -> float:
        """
        Perform a single training step using teacher forcing.

        Takes source embeddings and target text, and trains the model to predict
        the next token in the sequence given the previous tokens and source embeddings.

        Args:
            source_rep: List of lists of floats representing the source embeddings
                Shape: (src_seq_len, d_model)
            target_text: The target text string to predict

        Returns:
            The training loss for this step (float)
        """
        self.decoder.train()
        if not source_rep:
            return 0.0

        encoder_outputs = torch.tensor(source_rep, device=self.device).unsqueeze(1)

        target_tokens = self.tokenizer.encode(
            target_text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        target_tokens = target_tokens.squeeze(0)
        if target_tokens.size(0) < 2:
            return 0.0

        dec_input = target_tokens[:-1].unsqueeze(1)
        dec_target = target_tokens[1:].unsqueeze(1)

        seq_len = dec_input.size(0)
        tgt_mask = generate_subsequent_mask(seq_len, self.device)

        logits = self.decoder(encoder_outputs, dec_input, tgt_mask)
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        dec_target_flat = dec_target.view(-1)

        loss = self.criterion(logits_flat, dec_target_flat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate_text(self, source_rep: List[List[float]], max_length: int = 40) -> str:
        """
        Generate text from source embeddings using the trained model.

        This method uses autoregressive generation: at each step, it predicts the next
        token based on previously generated tokens and the source embeddings.

        Args:
            source_rep: List of lists of floats representing the source embeddings
                Shape: (src_seq_len, d_model)
            max_length: Maximum number of tokens to generate

        Returns:
            The generated text string
        """
        self.decoder.eval()
        if not source_rep:
            return ""
        encoder_outputs = torch.tensor(source_rep, device=self.device).unsqueeze(1)

        start_token_id = self.tokenizer.cls_token_id or 101
        current_input = torch.tensor([start_token_id], device=self.device).unsqueeze(1)

        generated_tokens = []
        for _ in range(max_length):
            seq_len = current_input.size(0)
            tgt_mask = generate_subsequent_mask(seq_len, self.device)
            logits = self.decoder(encoder_outputs, current_input, tgt_mask)
            logits_step = logits[-1, 0, :]
            next_token_id = torch.argmax(F.log_softmax(logits_step, dim=-1)).item()
            generated_tokens.append(next_token_id)

            next_token = torch.tensor([next_token_id], device=self.device).unsqueeze(1)
            current_input = torch.cat([current_input, next_token], dim=0)
            if next_token_id == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @torch._dynamo.disable
    def save_model(self, save_dir: str) -> None:
        """
        Saves the model + config + tokenizer.

        This method saves the model state, configuration, and tokenizer to disk.
        It disables torch.compile if you're on PyTorch 2.0, so state_dict() won't break.

        Args:
            save_dir: Directory path where to save the model

        Note:
            We no longer save optimizer state to avoid reference issues
            when loading/saving multiple times.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "reverser_seq2seq_state.pt")

        torch.save(
            {
                "decoder_state_dict": self.decoder.state_dict(),
                "config": self.config,
            },
            save_path,
        )

        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_path}")

    @torch._dynamo.disable
    def load_model(self, load_dir: str, device: Optional[str] = None) -> None:
        """
        Loads model + optimizer state into this *existing* instance.

        This method loads a previously saved model state into the current instance.
        It handles device mapping and configuration updates automatically.

        Args:
            load_dir: Directory path from which to load the model
            device: The device to use ('cuda', 'cpu', or None to auto-select)

        IMPORTANT:
          - This requires that your constructor used the same architecture
            hyperparameters (d_model, nhead, etc.) that were in the checkpoint.
          - If you want to load a different config from the checkpoint,
            see the alternative approach in the Appendix below.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        load_path = os.path.join(load_dir, "reverser_seq2seq_state.pt")
        checkpoint = torch.load(load_path, map_location=self.device)

        loaded_config = checkpoint.get("config", {})
        self.config.update(loaded_config)

        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)

        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self.decoder.to(self.device)

        self.optimizer = optim.AdamW(self.decoder.parameters(), lr=self.config["lr"])

        logger.info(f"Model successfully loaded from {load_dir}")
