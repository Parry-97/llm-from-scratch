# Python Project Best Practices: A Recap

This document summarizes the key concepts for structuring, packaging, and developing modern Python projects. We started with a common `ImportError` and ended with a complete, professional workflow.

### 1. The Core Problem: Understanding Python's Imports

The initial problem was an `ImportError` when trying to import code from a sibling directory (`gpt_architecture`) into `pretraining/utils.py`.

*   **The Cause**: Python's import system relies on a list of directories called `sys.path`. When you run `python pretraining/utils.py`, only the `pretraining/` directory is added to this path. Python has no knowledge of the sibling `gpt_architecture/` directory.

*   **The Solution**: Run your script as a **module**. From the project root, use the `-m` flag:
    ```bash
    # Correct way to run a script within a project
    uv run python -m pretraining.utils
    ```
    This command adds the **project root** to `sys.path`, making all sub-packages (like `pretraining` and `gpt_architecture`) visible to the interpreter.

### 2. The Ideal Project Structure: The `src` Layout

To prevent import problems and create a more robust project, the recommended structure is the `src` layout.

**Flat Layout (Less Ideal):**
```
my-project/
├── my_package/
├── tests/
└── pyproject.toml
```

**`src` Layout (Recommended):**
```
my-project/
├── src/
│   └── my_package/  <-- Your code lives here
│       ├── __init__.py
│       └── ...
├── tests/
└── pyproject.toml
```

**Key Benefits of the `src` Layout:**

1.  **Prevents Accidental Imports**: It makes it impossible to import your package from the project root accidentally. This forces you to install the package to use it, which mimics how end-users will interact with your code.
2.  **Forces Correct Testing**: You must install your package to run tests, ensuring your test suite runs against the *installed* code, not the local source files. This helps catch packaging bugs early.
3.  **Clean Separation**: It creates a clear distinction between your Python source code (`src/`) and project management files (`pyproject.toml`, `README.md`, etc.).

### 3. Packaging: Why We Need Build Systems

Even though Python is an interpreted language, you don't distribute your code by just sharing `.py` files. You distribute it in standardized packages. A **build system** is the tool that creates these packages.

*   **Wheel (`.whl`)**: The modern standard. This is a pre-built package that installs very quickly on a user's machine. Your build system's main job is to create this wheel file.
*   **Source Distribution (`sdist`)**: A compressed archive of your source code. This is a fallback that requires the user to have build tools on their machine to install.

The entire packaging process is configured in the `pyproject.toml` file.

### 4. Choosing Your Tools

*   **Package Manager (`uv`)**: An extremely fast tool for resolving and installing packages into your virtual environment. It reads your `pyproject.toml` to know what to install. It replaces `pip` and `venv`.

*   **Build System (`hatchling`)**: A modern, fast, and easy-to-use build system. It reads your `pyproject.toml` and source code to generate the final wheel package. To use it, you simply declare it in your `pyproject.toml`:
    ```toml
    # pyproject.toml
    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"
    ```

### 5. The Professional Workflow: Editable Installs

This concept ties everything together.

*   **What is it?**: An "editable" install is performed with `uv pip install -e .`. Instead of copying your code, it creates a **link** from your virtual environment's `site-packages` directory back to your source code.

*   **Why use it?**: It allows you to make changes to your source code and have them **instantly reflected** without needing to re-install the package after every edit.

**The Complete Workflow:**

1.  Structure your project using the `src` layout.
2.  Create a virtual environment (`uv venv`).
3.  Install your project in **editable mode** (`uv pip install -e .`).
4.  Now, you can run your scripts as modules (e.g., `uv run python -m my_package.main`) and run your tests. Any changes you make to your code in the `src` directory will be picked up immediately.

This workflow ensures that you are always developing and testing against your package in a way that is consistent with how it will be installed and used by others, leading to more reliable and maintainable software.
