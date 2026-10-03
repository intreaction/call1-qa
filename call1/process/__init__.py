"""Call1 Process: upload/import, the job-graph builder, the worker loop and the operator console.

Process reaches Store only through its HTTP API (``store_client``) with its own service key. It
never imports ``call1.store``, ``call1.db`` or ``call1.ingest`` and never opens a Store file;
``tests/test_split_boundaries.py`` enforces that. See ``README.md`` in this package.
"""

PROCESS_VERSION = "0.1.0"
