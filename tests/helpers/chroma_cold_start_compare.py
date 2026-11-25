"""Compare concurrent first-open behavior in disposable local index directories."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, Lock
from time import perf_counter
from uuid import uuid4

from ai_agent.retrieval import rag


def compare(root, output, repeats, arm, separate_indexes=False):
    original, guard, production_lock = rag.Chroma, Lock(), rag._chroma_init_lock

    def serialized(**kwargs):
        with guard:
            return original(**kwargs)

    try:
        arms = (("production", original),) if arm == "production" else (
            ("baseline", original), ("chroma-init-lock", serialized),
        )
        for name, constructor in arms:
            if arm != "both" and name != arm:
                continue
            rag.Chroma = constructor
            rag._chroma_init_lock = production_lock if name == "production" else nullcontext()
            for trial in range(repeats):
                path = str(output / name / str(trial))
                start = Barrier(4)

                def open_index(index):
                    start.wait(timeout=10)
                    try:
                        destination = str(Path(path) / str(index)) if separate_indexes else path
                        rag.LearningRAG(str(root), data_dir=destination)
                    except Exception as error:
                        return f"{type(error).__name__}: {error}"
                    return "ok"

                tick = perf_counter()
                with ThreadPoolExecutor(4) as pool:
                    outcomes = list(pool.map(open_index, range(4)))
                print(name, trial, round(perf_counter() - tick, 3), outcomes, flush=True)
    finally:
        rag.Chroma = original
        rag._chroma_init_lock = production_lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--arm", choices=("both", "baseline", "chroma-init-lock", "production"), default="both")
    parser.add_argument("--separate-indexes", action="store_true")
    args = parser.parse_args()
    compare(args.materials.resolve(), args.output / uuid4().hex, args.repeats, args.arm, args.separate_indexes)
