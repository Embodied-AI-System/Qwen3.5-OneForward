# Contributing

Thanks for helping improve OneForward.

## Before opening a pull request

1. Keep the project honest about its scope. Do not describe conditional candidate
   probabilities as calibrated correctness probabilities.
2. Add tests for behavior changes, especially tokenizer/template invariants and
   API response shapes.
3. Run:

   ```bash
   PYTHONPATH=. python -m unittest discover -s tests -v
   python -m compileall -q app.py core.py inference.py tests scripts
   ```

4. For model-quality or latency claims, include the model revision, hardware,
   prompt mode, dataset, sample count, and exact measurement procedure.
5. Never commit model weights, credentials, private prompts/data, or machine-local
   runtime files.

## Useful contribution areas

- controlled option-order and label-token bias evaluation
- calibration studies on public labeled datasets
- KV prefix caching and batch scheduling improvements
- additional model backends with explicit tokenizer-invariant tests
- Noul and Score research implementations with clearly documented semantics
- accessibility, documentation, and reproducibility improvements

By contributing, you agree that your contribution is licensed under Apache-2.0.
