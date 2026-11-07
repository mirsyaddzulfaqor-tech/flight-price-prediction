"""Tiny notebook builder + executor (no Jupyter needed).

Usage from a build script (keeps real outputs without needing Jupyter):
    from nbrun import Notebook
    nb = Notebook(workdir)
    nb.md("# Title")
    nb.code("x = 1\nx")
    nb.save("analysis.ipynb")   # executes every code cell in order, stores real outputs
"""
import ast
import base64
import contextlib
import io
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class Notebook:
    def __init__(self, workdir):
        self.workdir = workdir
        self.cells = []
        self.ns = {"__name__": "__main__"}
        self.count = 0

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {},
                           "source": text.strip("\n").splitlines(keepends=True)})

    def code(self, src):
        src = src.strip("\n")
        self.count += 1
        outputs = self._run(src)
        self.cells.append({"cell_type": "code", "execution_count": self.count,
                           "metadata": {}, "outputs": outputs,
                           "source": src.splitlines(keepends=True)})

    def _run(self, src):
        outputs = []
        buf = io.StringIO()
        cwd = os.getcwd()
        os.chdir(self.workdir)
        result = None
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<cell>", "exec"), self.ns)
                if last is not None:
                    result = eval(compile(last, "<cell>", "eval"), self.ns)
        except Exception:
            os.chdir(cwd)
            raise RuntimeError(f"Cell failed:\n{src}\n\n{traceback.format_exc()}")
        os.chdir(cwd)
        text = buf.getvalue()
        if text:
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                                     "text/plain": [f"<Figure {num}>"]}})
        plt.close("all")
        if result is not None and not isinstance(result, (matplotlib.artist.Artist,
                                                          matplotlib.figure.Figure)):
            data = {"text/plain": repr(result).splitlines(keepends=True)}
            if isinstance(result, (pd.DataFrame, pd.Series)):
                frame = result.to_frame() if isinstance(result, pd.Series) else result
                data["text/html"] = frame.to_html(max_rows=30).splitlines(keepends=True)
            outputs.append({"output_type": "execute_result", "execution_count": self.count,
                            "metadata": {}, "data": data})
        return outputs

    def save(self, path):
        nb = {"cells": self.cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
