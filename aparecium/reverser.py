"""
Seq2Seq Reverser Module

This module provides functionality for converting numeric representations 
back to text using a Transformer-based sequence-to-sequence architecture. 
It includes a decoder model that can be trained with teacher forcing and 
used at inference to generate text from embedded representations.

Classes:
    TransformerSeq2SeqModel: The neural network model (Transformer decoder).
    Seq2SeqReverser: Main interface for training and text generation.

Example:
    >>> from reverser import Seq2SeqReverser
    >>> reverser = Seq2SeqReverser()
    >>> loss = reverser.train_step_batch(
    ...     source_rep_batch=[[[0.1]*768]*10, [[0.2]*768]*12],  # example embeddings
    ...     target_text_batch=["Hello world", "Another example"]
    ... )
    >>> text_output = reverser.generate_text(
    ...     source_rep=[[0.1]*768]*10,
    ...     max_length=20
    ... )
    >>> print(text_output)
    "Hello world"
"""

from typing import Optional, List
import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn.functional as F  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch._dynamo  # type: ignore

from .logger import logger  # type: ignore
from .exceptions import ReverserError, ConfigurationError  # type: ignore


def generate_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) mask of shape (seq_len, seq_len).

    This mask ensures each position can only attend to previous positions,
    which is needed for autoregressive decoding. True values indicate positions
    that should be masked out (future tokens).

    Args:
        seq_len (int): Length of the sequence to mask.
        device (torch.device): The torch device on which to create the mask.

    Returns:
        torch.Tensor: A boolean tensor of shape (seq_len, seq_len).
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
        token_embedding (nn.Embedding): Embedding layer for input tokens.
        pos_embedding (nn.Embedding): Positional embedding layer.
        transformer_decoder (nn.TransformerDecoder): Stacked decoder layers.
        fc_out (nn.Linear): Projection layer from d_model to vocab size.
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
            vocab_size (int): Size of the output vocabulary.
            d_model (int): Dimensionality of embeddings and hidden states.
            num_decoder_layers (int): Number of Transformer decoder layers.
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimensionality of the feed-forward layers.
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
        Forward pass of the Transformer decoder model.

        Args:
            encoder_outputs (torch.Tensor):
                Output tensor (memory) from an encoder.
                Shape: (src_seq_len, batch_size, d_model)
            tgt_input_ids (torch.Tensor):
                Target input token IDs. Shape: (tgt_seq_len, batch_size)
            tgt_mask (torch.Tensor, optional):
                Autoregressive mask of shape (tgt_seq_len, tgt_seq_len)
                to block attention to future tokens.

        Returns:
            torch.Tensor:
                Logits for next-token prediction.
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

    Provides training (with teacher forcing) and inference methods,
    as well as model saving/loading functionality.
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
            model_name (str):
                The name or path of the Hugging Face tokenizer to use.
            d_model (int):
                Dimensionality of embeddings and hidden states.
            num_decoder_layers (int):
                Number of stacked transformer decoder layers.
            nhead (int):
                Number of attention heads.
            dim_feedforward (int):
                Dimensionality of the transformer's feed-forward networks.
            lr (float):
                Learning rate for the optimizer.
            device (Optional[str]):
                The device to use ('cuda', 'cpu', or None for auto-select).
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

        Args:
            source_rep (List[List[float]]):
                Source embeddings of shape (src_seq_len, d_model).
            target_text (str):
                The target text string to predict.

        Returns:
            float: The training loss for this step.
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
        Perform a batched teacher-forcing training step.

        Args:
            source_rep_batch (List[List[List[float]]]):
                A list of source embedding matrices, each (src_seq_len_i, d_model).
            target_text_batch (List[str]):
                A list of target strings corresponding to each source batch.
            max_source_length (int):
                Truncate source sequences to this length.
            max_target_length (int):
                Truncate target sequences to this length.

        Returns:
            float: The loss value for this batch.
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

        Args:
            source_rep (List[List[float]]):
                Source embeddings of shape (src_seq_len, d_model).
            max_length (int):
                Maximum number of tokens to generate.
            num_beams (int):
                Number of beams for beam search. If > 1, beam search is used.
            do_sample (bool):
                Whether to sample from the probability distribution (if num_beams=1).
            top_k (int):
                Top-k sampling filter. Used only if do_sample=True.
            top_p (float):
                Nucleus (top-p) sampling filter. Used only if do_sample=True.
            temperature (float):
                Softmax temperature for controlling randomness in sampling.

        Returns:
            str: The generated text, with special tokens removed.
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

        Args:
            encoder_outputs (torch.Tensor):
                Encoded source representations, shape (src_seq_len, 1, d_model).
            max_length (int):
                Maximum number of tokens to generate.
            do_sample (bool):
                If True, sample from the probability distribution;
                if False, use greedy decoding.
            top_k (int):
                Top-k sampling filter (only used if do_sample=True).
            top_p (float):
                Top-p (nucleus) sampling filter (only used if do_sample=True).
            temperature (float):
                Softmax temperature for controlling randomness.

        Returns:
            str: Generated text with special tokens removed.
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

        Args:
            encoder_outputs (torch.Tensor):
                Encoded source representations, shape (src_seq_len, 1, d_model).
            max_length (int):
                Maximum number of tokens to generate before stopping.
            num_beams (int):
                Number of beams (candidate sequences) to keep at each step.
            temperature (float):
                Softmax temperature for controlling randomness in the distribution.

        Returns:
            str: Generated text from the highest-scoring beam,
                 with special tokens removed.
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
        Sample a token ID from logits using top-k and/or top-p (nucleus) filtering.

        Args:
            logits (torch.Tensor):
                Raw logits of shape (vocab_size,).
            top_k (int):
                Only consider top-k tokens. If <= 0, disable top-k filtering.
            top_p (float):
                Nucleus filtering; only consider tokens with cumulative probability
                above this threshold in sorted order. Must be in (0, 1].

        Returns:
            int: Sampled token ID.

        Raises:
            ReverserError: If sampling fails due to invalid logits or parameters.
        """
        try:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            probs = F.softmax(logits, dim=-1)

            probs = torch.nan_to_num(probs, nan=0.0)
            probs = torch.clamp(probs, min=0.0)

            # Top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(
                    probs, min(top_k, probs.size(-1))
                )
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
                probs = torch.zeros_like(probs).scatter_(
                    -1, sorted_indices, sorted_probs
                )

            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = torch.ones_like(probs) / probs.size(-1)

            next_token_id = torch.multinomial(probs, 1).item()
            return next_token_id
        except Exception as e:
            logger.error(f"Failed to sample from logits: {str(e)}")
            raise ReverserError(f"Failed to sample from logits: {str(e)}")

    @torch._dynamo.disable
    def save_model(self, save_dir: str) -> None:
        """
        Save the model state and tokenizer to disk.

        Args:
            save_dir (str):
                Directory path where the model and config will be saved.

        Raises:
            ConfigurationError: If saving fails due to invalid directory or permissions.
            ReverserError: If model state saving fails.
        """
        try:
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
        except OSError as e:
            logger.error(f"Failed to create save directory or save model: {str(e)}")
            raise ConfigurationError(f"Failed to save model: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to save model state: {str(e)}")
            raise ReverserError(f"Failed to save model state: {str(e)}")

    @torch._dynamo.disable
    def load_model(self, load_dir: str, device: Optional[str] = None) -> None:
        """
        Load model and tokenizer states from disk into the current instance.

        Args:
            load_dir (str):
                Directory path where the model is saved.
            device (Optional[str]):
                Device to load the model on ('cuda', 'cpu', or None to auto-select).

        Raises:
            ConfigurationError: If loading fails due to invalid directory or file.
            ReverserError: If model state loading fails.
        """
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            logger.debug(f"Loading model on device: {device}")

            load_path = os.path.join(load_dir, "reverser_seq2seq_state.pt")
            if not os.path.exists(load_path):
                logger.error(f"Model file not found at {load_path}")
                raise ConfigurationError(f"Model file not found at {load_path}")

            checkpoint = torch.load(load_path, map_location=self.device)

            loaded_config = checkpoint.get("config", {})
            self.config.update(loaded_config)

            self.tokenizer = AutoTokenizer.from_pretrained(load_dir)

            self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

            self.decoder.to(self.device)

            self.optimizer = optim.AdamW(
                self.decoder.parameters(), lr=self.config["lr"]
            )

            logger.info(f"Model successfully loaded from {load_dir}")
        except OSError as e:
            logger.error(f"Failed to load model file: {str(e)}")
            raise ConfigurationError(f"Failed to load model file: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load model state: {str(e)}")
            raise ReverserError(f"Failed to load model state: {str(e)}")
