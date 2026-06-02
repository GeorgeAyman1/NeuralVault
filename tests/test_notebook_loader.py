import json

from core.ingestion.notebook_loader import NotebookLoader


def test_notebook_loader_extracts_cells_as_chunks(tmp_path):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Title\n\n",
                    "This is a meaningful markdown paragraph with enough detail.",
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    "print('hello world from a meaningful notebook code cell')",
                ],
            },
            {
                "cell_type": "markdown",
                "source": [],
            },
        ]
    }

    notebook_path = tmp_path / "test.ipynb"

    with open(notebook_path, "w", encoding="utf-8") as file:
        json.dump(notebook, file)

    loader = NotebookLoader()
    chunks = loader.load(str(notebook_path))

    assert len(chunks) == 2
    assert chunks[0]["text"] == "This is a meaningful markdown paragraph with enough detail."
    assert chunks[0]["metadata"]["cell_type"] == "markdown"
    assert chunks[0]["metadata"]["chunk_index"] == 0

    assert chunks[1]["text"] == "print('hello world from a meaningful notebook code cell')"
    assert chunks[1]["metadata"]["cell_type"] == "code"
