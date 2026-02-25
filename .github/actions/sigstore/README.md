This is our internally maintained version of https://github.com/sigstore/gh-action-sigstore-python (pinned to v3.0.0). We sync `action.py`, `templates/`, and `requirements.txt` from upstream. **DO NOT MODIFY THESE FILES DIRECTLY** — use the update script instead.

`action.yml` is maintained locally and must not be overwritten by upstream.

To update the code follow the steps below:
1. Run
    ```
    cd .github/actions/sigstore \
    && ./update-code.sh
    ```
    This will sync `action.py`, `templates/`, and `requirements.txt` from the upstream repository.
2. Thoroughly review the changed code to make sure it meets our standards and does not introduce any security vulnerabilities.
3. Verify that `action.yml` does not reference any external actions (e.g. `softprops/action-gh-release` must remain `./.github/actions/gh-release`).
