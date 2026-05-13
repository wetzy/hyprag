"""
benchmarks.queries
~~~~~~~~~~~~~~~~~~
Hand-designed hierarchical retrieval queries over the CPython standard library.

Ground-truth prefixes use the fully-qualified module paths that the chunker
now produces via chunk_directory() (e.g. ``asyncio.base_events``, not just
``base_events``).  A chunk is considered correct if its node_path equals or
starts with one of the listed prefixes.

Notes on individual queries
---------------------------
- collections.OrderedDict   : C-implemented in CPython 3.12+; no Python AST
  node exists, so this query tests whether retrieval falls back gracefully.
- argparse._SubParsersAction : CPython 3.14 uses new syntax; parses correctly
  on Python >=3.14 (the intended target), returns 0 hits on older runtimes.
"""

from __future__ import annotations
from dataclasses import dataclass, field

__all__ = ["HierarchicalQuery", "QUERIES"]


@dataclass
class HierarchicalQuery:
    text: str
    ground_truth_prefixes: list[str]
    notes: str = ""


QUERIES: list[HierarchicalQuery] = [
    HierarchicalQuery(
        text="how does asyncio schedule callbacks",
        ground_truth_prefixes=["asyncio.base_events.BaseEventLoop"],
        notes="call_soon/call_later/call_at all live on BaseEventLoop in asyncio/base_events.py",
    ),
    HierarchicalQuery(
        text="where is SSL certificate validation",
        ground_truth_prefixes=["ssl.SSLContext", "ssl.SSLSocket"],
        notes="check_hostname, verify_mode, match_hostname — class subtree in ssl.py",
    ),
    HierarchicalQuery(
        text="what does the csv DictWriter do",
        ground_truth_prefixes=["csv.DictWriter"],
        notes="writeheader/writerow/writerows — single class in csv.py",
    ),
    HierarchicalQuery(
        text="how is os path join implemented across platforms",
        ground_truth_prefixes=["posixpath.join", "ntpath.join"],
        notes="Two parallel implementations; tests cross-module retrieval",
    ),
    HierarchicalQuery(
        text="where are HTTP status codes defined",
        ground_truth_prefixes=["http.HTTPStatus"],
        notes="http/__init__.py; HTTPStatus is an IntEnum with ~60 members",
    ),
    HierarchicalQuery(
        text="how does threading Lock work internally",
        ground_truth_prefixes=["threading._RLock", "threading.Condition", "threading.Lock"],
        notes="Lock is a factory; _RLock and Condition carry the real logic",
    ),
    HierarchicalQuery(
        text="how does pickle handle custom classes",
        ground_truth_prefixes=["pickle._Pickler", "pickle._Unpickler", "pickle.PickleError"],
        notes="reduce_ex, save_reduce, persistent_id — class subtree",
    ),
    HierarchicalQuery(
        text="where is JSON parsing logic",
        ground_truth_prefixes=["json.decoder.JSONDecoder"],
        notes="json/decoder.py; JSONDecoder.decode + raw_decode + py_scanstring",
    ),
    HierarchicalQuery(
        text="how does logging configure handlers",
        ground_truth_prefixes=["logging.handlers", "logging.Logger"],
        notes="logging/handlers.py (BaseRotatingHandler etc.) + Logger in __init__",
    ),
    HierarchicalQuery(
        text="what does collections OrderedDict do",
        ground_truth_prefixes=["collections.OrderedDict"],
        notes="C-implemented in 3.12+; intentionally hard query — tests graceful fallback",
    ),
    HierarchicalQuery(
        text="how does urllib parse URLs",
        ground_truth_prefixes=["urllib.parse"],
        notes="urllib/parse.py; urlparse, urlsplit, urljoin, urldefrag",
    ),
    HierarchicalQuery(
        text="how does subprocess Popen launch processes",
        ground_truth_prefixes=["subprocess.Popen"],
        notes="subprocess.py; __init__, _execute_child, communicate, wait",
    ),
    HierarchicalQuery(
        text="how does datetime parse strings",
        ground_truth_prefixes=["_pydatetime.datetime", "_strptime"],
        notes="_pydatetime.py has the datetime class; _strptime.py handles strptime",
    ),
    HierarchicalQuery(
        text="how does argparse handle subparsers",
        ground_truth_prefixes=["argparse._SubParsersAction", "argparse.ArgumentParser"],
        notes="argparse.py; Python 3.14 syntax — parses on >=3.14 only",
    ),
    HierarchicalQuery(
        text="where is base64 encoding implemented",
        ground_truth_prefixes=["base64"],
        notes="base64.py; b64encode, b64decode, urlsafe variants — worked well already",
    ),
    HierarchicalQuery(
        text="how does sqlite3 connect to a database",
        ground_truth_prefixes=["sqlite3.dbapi2"],
        notes="sqlite3/dbapi2.py has connect(); sqlite3/__init__.py re-exports it",
    ),
    HierarchicalQuery(
        text="how does heapq maintain the heap invariant",
        ground_truth_prefixes=["heapq"],
        notes="heapq.py; heappush/heappop/_siftdown/_siftup — got 1.00 recall already",
    ),
    HierarchicalQuery(
        text="what does the abc module provide",
        ground_truth_prefixes=["abc.ABCMeta", "abc.abstractmethod", "_py_abc.ABCMeta"],
        notes="abc.py + _py_abc.py; two parallel implementations (pure Python + C)",
    ),
    HierarchicalQuery(
        text="how does xml etree parse documents",
        ground_truth_prefixes=["xml.etree.ElementTree", "xml.etree.ElementPath"],
        notes="xml/etree/ElementTree.py + ElementPath.py",
    ),
    HierarchicalQuery(
        text="how does multiprocessing share memory between processes",
        ground_truth_prefixes=[
            "multiprocessing.shared_memory",
            "multiprocessing.managers",
            "multiprocessing.sharedctypes",
        ],
        notes="Three submodules; tests multi-module subtree expansion",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_relevant(node_path: str, prefixes: list[str]) -> bool:
    for p in prefixes:
        if node_path == p or node_path.startswith(p + "."):
            return True
    return False


def relevant_chunks(chunks, prefixes: list[str]) -> list:
    return [c for c in chunks if is_relevant(c.node_path, prefixes)]
