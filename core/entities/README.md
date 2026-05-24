# Entity Runtime Notes

The entity package uses Cython extensions for `trie` and `co_occurence`.
Build them for the active Python runtime before running entity-level tests:

```bash
cd core/entities
python _cython_setup.py build_ext --inplace
```

The generated `.so` files are local build artifacts and are ignored by git.

Phase 2 domain stabilization smoke test:

```bash
PYTHONPATH=core python -m pytest -q tests/entities/test_phase2_domain_stabilization.py
```
