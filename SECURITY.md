# Security policy

Please report security issues privately through GitHub Security Advisories.
Do not open a public issue containing credentials, private dataset paths, or
malicious checkpoint samples.

PyTorch checkpoints may contain pickle data. Use only the release asset whose
SHA-256 matches `checkpoints/README.md`; the `--legacy-e266` loader refuses any
other file before unpickling it. Do not load checkpoints from untrusted sources.
