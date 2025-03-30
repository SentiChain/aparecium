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

    def train_step_batch(
        self,
        source_rep_batch: List[List[List[float]]],
        target_text_batch: List[str],
        max_source_length: int = 256,
        max_target_length: int = 256,
    ) -> float:
        """
        Batched teacher-forcing training step.

        Args:
            source_rep_batch: A list of source embedding matrices,
                each shape: (src_seq_len_i, d_model). Typically length = batch_size.
            target_text_batch: A list of target strings of length = batch_size.
            max_source_length: Optional limit on source sequence length (truncate).
            max_target_length: Optional limit on target sequence length (truncate).

        Returns:
            The loss value (float) for this batch.
        """
        self.decoder.train()
        batch_size = len(source_rep_batch)
        if batch_size == 0:
            return 0.0

        src_tensors = []
        for rep in source_rep_batch:
            rep = rep[:max_source_length]
            t = torch.tensor(rep, dtype=torch.float32, device=self.device)
            src_tensors.append(t)

        encoder_outputs = torch.nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=False
        )

        encoded_targets = self.tokenizer(
            target_text_batch,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        target_tokens = encoded_targets["input_ids"].to(self.device)

        if target_tokens.size(1) < 2:
            return 0.0

        dec_input = target_tokens[:, :-1]
        dec_target = target_tokens[:, 1:]

        dec_input = dec_input.transpose(0, 1)  # (tgt_seq_len-1, batch_size)
        dec_target = dec_target.transpose(0, 1)  # (tgt_seq_len-1, batch_size)

        seq_len = dec_input.size(0)
        tgt_mask = generate_subsequent_mask(seq_len, self.device)

        logits = self.decoder(encoder_outputs, dec_input, tgt_mask)
        vocab_size = logits.size(-1)

        loss = self.criterion(
            logits.view(-1, vocab_size),
            dec_target.reshape(-1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def generate_text(
        self,
        source_rep: List[List[float]],
        max_length: int = 40,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text from source embeddings using beam search, greedy decoding, or sampling.

        This method takes encoded vector representations and generates human-readable text
        by decoding them through the sequence-to-sequence model. It supports multiple decoding
        strategies to balance between deterministic outputs and creative diversity.

        Args:
            source_rep: Source embeddings matrix with shape (src_seq_len, d_model).
                Each row represents a token's embedding in the source sequence.
            max_length: Maximum number of tokens to generate before stopping.
            num_beams: Number of beams for beam search. Values > 1 activate beam search,
                which performs a breadth-first search through the probability space.
            do_sample: Whether to sample from the probability distribution (True) or
                use greedy decoding (False). Only applies when num_beams=1.
            top_k: In sampling mode, only consider the top-k most probable tokens.
            top_p: In sampling mode, only consider tokens within the top-p probability mass
                (nucleus sampling).
            temperature: Controls randomness in sampling. Higher values (e.g., 1.5)
                produce more diverse outputs, lower values (e.g., 0.7) produce more
                deterministic outputs. Range: (0.0, inf).

        Returns:
            A string containing the generated text with special tokens removed.

        Note:
            For maximum determinism, use greedy decoding (num_beams=1, do_sample=False).
            For maximum diversity, use sampling with higher temperature values.
            For a balance of quality and diversity, beam search (num_beams=3-5) often works well.
        """
        self.decoder.eval()
        if not source_rep:
            return ""

        encoder_outputs = torch.tensor(source_rep, device=self.device).unsqueeze(1)

        # Beam search with num_beams > 1
        if num_beams > 1:
            return self._beam_search(
                encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
            )
        else:
            # Greedy or sampling decode
            return self._sample_or_greedy_decode(
                encoder_outputs,
                max_length=max_length,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

    def _sample_or_greedy_decode(
        self,
        encoder_outputs: torch.Tensor,
        max_length: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> str:
        """
        Perform autoregressive text generation using either greedy decoding or sampling.

        This internal method implements the core text generation loop for non-beam search
        approaches. It iteratively builds the output sequence token-by-token until reaching
        a stop condition (max length or end token).

        Args:
            encoder_outputs: Tensor containing encoder representations with shape
                (src_seq_len, batch_size=1, d_model).
            max_length: Maximum number of tokens to generate.
            do_sample: If True, sample from the probability distribution;
                if False, perform greedy decoding (take the most probable token).
            top_k: In sampling mode, only consider the top-k most probable tokens.
            top_p: In sampling mode, only consider tokens within the top-p probability mass
                (nucleus sampling).
            temperature: Softmax temperature for controlling randomness. Lower values make
                the distribution more peaked, higher values make it more uniform.

        Returns:
            A string containing the generated text with special tokens removed.

        Note:
            This function handles both deterministic (greedy) and stochastic (sampling)
            decoding based on the do_sample parameter.
        """
        start_token_id = self.tokenizer.cls_token_id or 101
        sep_token_id = self.tokenizer.sep_token_id or 102

        current_input = torch.tensor([start_token_id], device=self.device).unsqueeze(1)
        generated_tokens = []

        for _ in range(max_length):
            seq_len = current_input.size(0)
            tgt_mask = generate_subsequent_mask(seq_len, self.device)
            logits = self.decoder(encoder_outputs, current_input, tgt_mask)
            logits_step = logits[-1, 0, :]  # Shape: (vocab_size,)

            # Apply temperature
            logits_step = logits_step / max(temperature, 1e-8)

            if do_sample:
                # Top-k or nucleus sampling
                next_token_id = self._sample_from_logits(
                    logits_step, top_k=top_k, top_p=top_p
                )
            else:
                # Greedy decoding
                next_token_id = torch.argmax(logits_step, dim=-1).item()

            generated_tokens.append(next_token_id)

            next_token = torch.tensor([next_token_id], device=self.device).unsqueeze(1)
            current_input = torch.cat([current_input, next_token], dim=0)

            if next_token_id == sep_token_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _beam_search(
        self,
        encoder_outputs: torch.Tensor,
        max_length: int,
        num_beams: int,
        temperature: float,
    ) -> str:
        """
        Implement beam search decoding for more optimal text generation.

        Beam search maintains multiple candidate sequences at each step, expanding
        each by considering the top-N next tokens, then keeping only the overall
        top-N sequences to continue with. This reduces the risk of getting stuck
        in suboptimal paths that greedy decoding might fall into.

        Args:
            encoder_outputs: Tensor containing encoder representations with shape
                (src_seq_len, batch_size=1, d_model).
            max_length: Maximum number of tokens to generate before stopping.
            num_beams: Number of beams (candidate sequences) to maintain at each step.
                Higher values provide more thorough search but increase computation.
            temperature: Temperature for softmax over logits. Lower values make the
                distribution more peaked, higher values make it more uniform.

        Returns:
            A string containing the generated text from the highest-scoring beam,
            with special tokens removed.

        Note:
            This is a simple beam search implementation that tracks log probabilities
            and handles early stopping when sequences are complete.
        """
        start_token_id = self.tokenizer.cls_token_id or 101
        sep_token_id = self.tokenizer.sep_token_id or 102

        beams = [
            (
                torch.tensor([start_token_id], device=self.device).unsqueeze(1),
                0.0,
            )
        ]

        for _ in range(max_length):
            new_beams = []
            for tokens, log_prob in beams:
                if tokens[-1].item() == sep_token_id:
                    new_beams.append((tokens, log_prob))
                    continue

                seq_len = tokens.size(0)
                tgt_mask = generate_subsequent_mask(seq_len, self.device)
                logits = self.decoder(encoder_outputs, tokens, tgt_mask)
                logits_step = logits[-1, 0, :] / max(temperature, 1e-8)

                probs = F.log_softmax(logits_step, dim=-1)
                top_probs, top_ids = probs.topk(num_beams)

                for i in range(num_beams):
                    next_id = top_ids[i].item()
                    next_score = top_probs[i].item()
                    new_tokens = torch.cat(
                        [tokens, torch.tensor([[next_id]], device=self.device)], dim=0
                    )
                    new_beams.append((new_tokens, log_prob + next_score))

            new_beams.sort(key=lambda b: b[1], reverse=True)
            beams = new_beams[:num_beams]

            all_finished = all(b[0][-1].item() == sep_token_id for b in beams)
            if all_finished:
                break

        best_tokens, best_log_prob = max(beams, key=lambda b: b[1])
        return self.tokenizer.decode(
            best_tokens.squeeze(1).tolist(), skip_special_tokens=True
        )

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> int:
        """
        Sample a token from a distribution of logits using top-k and/or nucleus (top-p) filtering.

        This method implements controlled sampling strategies to balance diversity and quality:
        1. Top-K filtering: Only consider the top-k most likely tokens
        2. Nucleus (Top-p) filtering: Only consider tokens comprising the top-p probability mass

        The method handles numerical stability issues and ensures a valid probability distribution
        before sampling.

        Args:
            logits: Raw logits tensor from the model with shape (vocab_size,)
            top_k: Only consider the top-k tokens with highest probability. If <= 0,
                all tokens are considered.
            top_p: Only consider tokens whose cumulative probability exceeds this threshold.
                Must be in range (0.0, 1.0]. If 1.0, all tokens are considered.

        Returns:
            An integer token ID sampled from the filtered distribution.

        Note:
            This implementation includes safeguards against numerical instabilities
            like NaN, infinity, or zero-sum probability distributions.
        """
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        probs = F.softmax(logits, dim=-1)

        probs = torch.nan_to_num(probs, nan=0.0)
        probs = torch.clamp(probs, min=0.0)

        # Top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
            kth_value = top_k_values[-1].clone()
            probs[probs < kth_value] = 0.0

        # Nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            if sorted_mask.any():
                first_idx = torch.where(cumulative_probs > top_p)[0][0].item()
                sorted_mask[first_idx] = False
            sorted_probs = sorted_probs * (~sorted_mask).float()
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = torch.ones_like(probs) / probs.size(-1)

        next_token_id = torch.multinomial(probs, 1).item()
        return next_token_id

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
