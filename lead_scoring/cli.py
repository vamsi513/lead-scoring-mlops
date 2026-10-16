"""Deterministic process exit for the batch entry points.

Why this exists
---------------
With pyarrow installed, a batch script that writes Parquet and then returns
normally can hang forever *after* it has finished. The work is done, the files
are closed and the output is printed; the process then blocks in libarrow's
global thread-pool destructor during C++ static finalisation, which runs after
Python has already torn down. Observed stack, from ``sample`` on the hung pid:

    exit
      __cxa_finalize_ranges
        std::shared_ptr<arrow::internal::ThreadPool>::~shared_ptr
          arrow::internal::ThreadPool::~ThreadPool
            arrow::internal::ThreadPool::Shutdown
              std::condition_variable::wait        <- never returns

Measured on this project: reproducible on every pyarrow release from 18.1.0
through 25.0.1, on macOS/arm64 with CPython 3.12. It fires when the entry point
is run as ``python -m data.generate`` from the repository root, and not when the
same ``main()`` is called from a script elsewhere on the path, so it depends on
module-finalisation order rather than on anything the pipeline does.

There is no public API to shut Arrow's global pools down early, so instead of
relying on static destructor ordering the entry points end the process
themselves. By the time ``run`` is called every Parquet file is closed and every
context manager has exited, so nothing is lost; ``os._exit`` only skips the
``atexit`` handlers, and this project registers none that matter. Each CLI does
its own cleanup explicitly (``session_scope`` stops Spark, the MLflow run is
closed by its context manager) rather than leaving it to interpreter shutdown.

If a future pyarrow fixes the deadlock, this module can be deleted and the
entry points can go back to ``raise SystemExit(main())``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable


def run(main: Callable[[list[str] | None], int], argv: list[str] | None = None) -> None:
    """Call ``main``, flush output, then end the process without unwinding.

    Never returns. Exceptions are allowed to propagate so that failures still
    print a traceback and produce a non-zero exit status.
    """
    code = main(argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
