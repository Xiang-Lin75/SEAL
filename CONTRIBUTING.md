# Contributing

Bug reports and focused pull requests are welcome. Before submitting code:

1. run `python -m compileall -q .`;
2. run `python -m unittest discover -s tests -v`;
3. do not commit datasets, checkpoints, TensorBoard logs, local absolute paths,
   credentials, or third-party papers;
4. document any change to metric definitions, dataset splits, model parameters,
   or MAC-counting protocol.

Scientific-result changes must include the resolved configuration, checkpoint
digest, ordered evaluation keys, per-utterance metrics, and aggregation code.
