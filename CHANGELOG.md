# Changelog

## Matrix reconstruction

- Reconstruct short English content from complete token states produced by the pinned `sentence-transformers/all-mpnet-base-v2` encoder and tokenizer.
- Provide a nonlinear projection into the original frozen MPNet vocabulary, with portable tensor weights and explicit encoder provenance.
- Expose single and batch matrix APIs, stored-matrix command-line input, and an HTTP service with the same validated input contract.
- Return reconstructed text, predicted interior token IDs, and visible input-row counts. Text convenience methods create compatible states without pooling.
- Require finite values, explicit framing, and valid right-padding masks. Reject incompatible pooled vectors and unsupported devices.
- Document source-fidelity measurements, financial and timing errors, training-source acknowledgments, and the separate terms for the learned projection and upstream vocabulary.

See [input compatibility](README.md#input-compatibility) for the matrix contract and integration changes.
