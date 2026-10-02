# Releasing nirnay to PyPI

## One-time setup

1. **Create accounts** on [pypi.org](https://pypi.org/account/register/) and
   [test.pypi.org](https://test.pypi.org/account/register/) (separate accounts). Turn on
   two-factor authentication on both (PyPI requires it to upload).
2. **Create API tokens.** On each site: Account settings → API tokens → *Add API token*, scope
   "Entire account" for the very first upload (you can narrow it to the `nirnay` project after).
   Copy each token; it starts with `pypi-` and is shown only once.
3. **Install the tools:** `pip install -e ".[dev]"` (includes `build` and `twine`).

## Before every release

1. **Fill in the project links** in `pyproject.toml`: uncomment `[project.urls]` and replace
   `YOUR-GITHUB-USERNAME`.
2. **Make the README images work on PyPI.** PyPI can't show relative paths like
   `assets/benchmarks/...png`. Replace them with absolute links, e.g.
   `https://raw.githubusercontent.com/YOUR-GITHUB-USERNAME/nirnay/main/assets/benchmarks/01_accuracy_with_descriptions.png`
   (push the repo to GitHub first so those URLs exist).
3. **Set the version** in both `pyproject.toml` (`version = "0.1.0"`) and
   `src/nirnay/__init__.py` (`__version__`). PyPI never lets you re-upload the same version.
4. **Check everything passes:**

   ```bash
   pytest
   ruff check .
   ```

5. **Build** (delete any old builds first):

   ```bash
   rm -rf dist
   python -m build
   twine check dist/*
   ```

6. **Check what you're about to publish.** It should contain only `src/nirnay`, `tests`,
   `README.md`, `LICENSE` and `pyproject.toml`. No notes, data, `.env` or drafts:

   ```bash
   tar -tzf dist/nirnay-0.1.0.tar.gz
   ```

## Upload to TestPyPI first

```bash
twine upload --repository testpypi dist/*
```

When asked, the username is `__token__` and the password is your **TestPyPI** token. Then check
it installs in a clean environment:

```bash
python -m venv /tmp/nirnay-test
source /tmp/nirnay-test/bin/activate        # Windows: \tmp\nirnay-test\Scripts\activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nirnay
python -c "import nirnay; print(nirnay.__version__)"
```

## Upload to PyPI

```bash
twine upload dist/*
```

Username `__token__`, password your **PyPI** token. The project page appears at
`https://pypi.org/project/nirnay/` within a minute; `pip install nirnay` then works anywhere.

## After the release

1. Tag it in git: `git tag v0.1.0 && git push origin v0.1.0`, and create a GitHub release.
2. Narrow your PyPI token's scope to just the `nirnay` project.
3. Optional: set up **Trusted Publishing** (PyPI → project → Publishing → add a GitHub Actions
   publisher) so future releases upload from GitHub Actions without a stored token.
