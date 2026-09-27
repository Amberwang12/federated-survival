# Releasing the package

Publish stable releases to PyPI so that users can install the package with
`python -m pip install federated-survival`. Keep the source repository on
GitHub for version control, issue tracking, documentation, and tagged source
releases.

## What is and is not released

The wheel contains the `federated_survival` Python package, its CLI entry
point, and packaged examples. The source distribution additionally contains
documentation and tests. Generated `results/`, local virtual environments,
IDE settings, caches, private datasets, and validation XML reports are not
release inputs.

Tests should remain in the source repository: they verify the public contract
and are not precomputed user results.

## Release checklist

1. Confirm that the version in `pyproject.toml` and
   `federated_survival/__init__.py` is new and identical.
2. Run the complete test suite.
3. Build a source distribution and wheel with `build`.
4. Validate both archives with `twine check --strict`.
5. Install the wheel into a fresh environment and run
   `federated-survival doctor`.
6. Optionally upload to TestPyPI and test installation there.
7. Upload the unchanged, validated archives to production PyPI.
8. Push the source revision to GitHub, then push the version tag. The
   `Release` workflow rebuilds the archives from the tagged revision and
   publishes them as a GitHub Release, so released assets are never taken from
   a local `dist/` directory.

```bash
python -m pip install --upgrade build twine
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

TestPyPI is separate from production PyPI:

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ federated-survival==0.7.0
```

When that installation is verified, publish to production:

```bash
python -m twine upload dist/*
```

Use a scoped PyPI API token and never commit it to the repository. GitHub
Actions Trusted Publishing is preferable once configured because it avoids a
long-lived upload token.

PyPI does not permit replacing files for an existing version. If anything in
an uploaded release must change, increment the package version, rebuild, and
upload the new version.

## GitHub Releases

`.github/workflows/release.yml` runs when a tag matching `v*` is pushed. It
rebuilds the source distribution and wheel from the tagged revision, validates
them with `twine check --strict`, checks that the tag matches the version in
`pyproject.toml`, writes `SHA256SUMS`, and then publishes a GitHub Release with
all three files attached.

```bash
git tag -a v0.7.0 -m "federated-survival 0.7.0"
git push origin v0.7.0
```

Release notes come from `.github/releases/<tag>.md` when that file exists;
write one for each release. If no file exists for the tag, the workflow falls
back to GitHub's commit-generated notes. The workflow declares the
`contents: write` permission it needs, so no extra secret is required.

Tag a revision only after CI is green for it. The release job packages and
publishes; it does not repeat the test matrix.

Official references:

- [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)
- [PyPI project page](https://pypi.org/project/federated-survival/)
- [GitHub repository](https://github.com/Amberwang12/federated-survival)
