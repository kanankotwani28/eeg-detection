"""Only src/eval/external.py may read Mumtaz paths. Checked statically on the import graph and on
string constants (docstrings/comments excluded)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED_IMPORTERS = {"src/eval/external.py"}
# files allowed to *mention* mumtaz in code: the reader itself, its parser module, and the synthetic generator
ALLOWED_MENTION = {"src/eval/external.py", "src/data/mumtaz.py", "src/data/synthetic.py"}


def _modules():
    for p in SRC.rglob("*.py"):
        yield p.relative_to(SRC.parent).as_posix(), ast.parse(p.read_text())


def _docstrings(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


def test_only_external_imports_mumtaz_module():
    bad = []
    for rel, tree in _modules():
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            if any(n.startswith("src.data.mumtaz") for n in names) and rel not in ALLOWED_IMPORTERS:
                bad.append(rel)
    assert not bad, f"modules importing src.data.mumtaz outside external.py: {bad}"


TRAINING_PACKAGES = ("src/train/", "src/fl/", "src/privacy/", "src/models/", "src/features/", "src/preprocess/")


def test_no_mumtaz_data_access_outside_external():
    """The config key, the path variable and the cache dataset name may appear only in external.py."""
    bad = []
    for rel, tree in _modules():
        if rel in ALLOWED_MENTION:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v.lower() == "mumtaz" or "MUMTAZ_PATH" in v or v.startswith("data/mumtaz"):
                    bad.append((rel, v))
    assert not bad, f"Mumtaz data access outside src/eval/external.py: {bad}"


def test_training_code_never_mentions_mumtaz():
    bad = []
    for rel, tree in _modules():
        if not rel.startswith(TRAINING_PACKAGES):
            continue
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if "mumtaz" in node.value.lower():
                    bad.append((rel, node.value[:60]))
            if isinstance(node, ast.Name | ast.Attribute):
                n = node.id if isinstance(node, ast.Name) else node.attr
                if "mumtaz" in n.lower():
                    bad.append((rel, n))
    assert not bad, f"training code references Mumtaz: {bad}"


def test_external_logs_access():
    text = (SRC / "eval" / "external.py").read_text()
    assert text.count("access_log(") >= 5
