This is our internally maintained version of https://github.com/softprops/action-gh-release (pinned to v2.5.0). We only accept changes to the `src` directory. **DO NOT TAKE CHANGES FROM THE DIST DIRECTORY.**

To update the code follow the steps below:
1. Run
    ```
    cd .github/actions/gh-release \
    && ./update-code.sh
    ```
    This will change the files in the `src` directory that have changed in the original repository.
2. Create the `dist` directory by running
    ```
    cd .github/actions/gh-release \
    && nvm use v20 \
    && yarn \
    && yarn build \
    && yarn package \
    && rm -fr node_modules \
    && rm -fr lib
    ```
3. Thoroughly review the changed code to make sure it meets our standards and does not introduce any security vulnerabilities.
