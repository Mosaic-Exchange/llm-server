"""
Parallel generation demo for the mosaicAI middleware (port 4000).
Submits two prompts at the same time and shows both responses side by side
as they stream in.
"""

import sys
import uuid
import queue
import threading
import shutil
import requests

MIDDLEWARE_URL = "http://127.0.0.1:4000"
LLAMA_URL = "http://127.0.0.1:8080"
MAX_LOADED = 2
SEP = "  │  "  # column separator


def get_registered_adapters() -> list[dict]:
    r = requests.get(f"{MIDDLEWARE_URL}/v1/adapters", timeout=5)
    r.raise_for_status()
    return r.json().get("data", [])


def get_active_adapters() -> list[dict]:
    try:
        r = requests.get(f"{LLAMA_URL}/lora-adapters", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def short_name(filename: str) -> str:
    name = filename.split("/")[-1]
    return name[:-5] if name.endswith(".gguf") else name


def print_adapter_table(registered: list[dict], active: list[dict]) -> None:
    active_paths = {a.get("path", "") for a in active}
    pct = int(len(active) / MAX_LOADED * 100)
    print(f"\n  Hot-loaded: {len(active)}/{MAX_LOADED} ({pct}% capacity)\n")
    print(f"  {'#':>2}  {'Adapter':<35}  Status")
    print(f"  {'--':>2}  {'-'*35}  ------")
    print(f"  {'0':>2}  {'(no adapter)':<35}")
    for i, adp in enumerate(registered, start=1):
        name = short_name(adp["adapter_filename"])
        hot = any(adp["adapter_filename"] in p or name in p for p in active_paths)
        print(f"  {i:>2}  {name:<35}  {'● loaded' if hot else ''}")
    print()


def _pick_adapter(registered: list[dict], label: str) -> tuple[str | None, str]:
    while True:
        raw = input(f"  [{label}] Select adapter (0 = none): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            sys.exit(0)
        if raw.isdigit() and 0 <= int(raw) <= len(registered):
            choice = int(raw)
            if choice == 0:
                return None, "base model"
            adp = registered[choice - 1]
            return adp["adapter_id"], short_name(adp["adapter_filename"])
        print(f"  Please enter a number between 0 and {len(registered)}.")


def _stream_into_queue(message: str, adapter_id: str | None, q: "queue.Queue[str | None]") -> None:
    payload = {"message": message, "request_id": uuid.uuid4().hex[:8]}
    if adapter_id:
        payload["adapter_id"] = adapter_id
    try:
        r = requests.post(
            f"{MIDDLEWARE_URL}/v1/generations/stream",
            json=payload, stream=True, timeout=120,
        )
        if r.status_code != 200:
            try:
                err = r.json().get("error", {})
                q.put(f"\n[ERROR {r.status_code}] {err.get('message', r.text)}")
            except Exception:
                q.put(f"\n[ERROR {r.status_code}] {r.text}")
        else:
            for chunk in r.iter_content(chunk_size=None):
                q.put(chunk.decode("utf-8"))
    except Exception as exc:
        q.put(f"\n[ERROR] {exc}")
    finally:
        q.put(None)  # sentinel so the reader knows we're done


class ColBuffer:
    """
    Accumulates raw streaming text and exposes it as display-ready lines
    word-wrapped to a fixed column width.
    """

    def __init__(self, width: int) -> None:
        self._width = width
        self._complete: list[str] = []
        self._partial: str = ""
        self.done: bool = False

    def feed(self, text: str) -> None:
        rest = self._partial + text
        while "\n" in rest:
            line, rest = rest.split("\n", 1)
            self._push(line)
        while len(rest) >= self._width:
            self._push(rest[:self._width])
            rest = rest[self._width:]
        self._partial = rest

    def _push(self, line: str) -> None:
        while len(line) > self._width:
            self._complete.append(line[:self._width])
            line = line[self._width:]
        self._complete.append(line)

    def finalize(self) -> None:
        if self._partial:
            self._push(self._partial)
            self._partial = ""
        self.done = True

    def pop(self) -> str | None:
        return self._complete.pop(0) if self._complete else None

    @property
    def has_line(self) -> bool:
        return bool(self._complete)


def _render_parallel(
    msg_a: str, aid_a: str | None,
    msg_b: str, aid_b: str | None,
    col_w: int,
) -> None:
    q_a: "queue.Queue[str | None]" = queue.Queue()
    q_b: "queue.Queue[str | None]" = queue.Queue()
    barrier = threading.Barrier(2)

    def worker(msg, aid, q):
        barrier.wait()  # sync so both requests go out at roughly the same time
        _stream_into_queue(msg, aid, q)

    threading.Thread(target=worker, args=(msg_a, aid_a, q_a), daemon=True).start()
    threading.Thread(target=worker, args=(msg_b, aid_b, q_b), daemon=True).start()

    buf_a = ColBuffer(col_w)
    buf_b = ColBuffer(col_w)

    def drain(buf: ColBuffer, q: "queue.Queue[str | None]") -> None:
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                buf.finalize()
            else:
                buf.feed(item)

    def print_rows() -> None:
        while buf_a.has_line or buf_b.has_line:
            la = buf_a.pop() or ""
            lb = buf_b.pop() or ""
            print(la.ljust(col_w) + SEP + lb)

    print()

    while not (buf_a.done and buf_b.done):
        drain(buf_a, q_a)
        drain(buf_b, q_b)
        print_rows()

        if not (buf_a.done and buf_b.done):
            for buf, q in [(buf_a, q_a), (buf_b, q_b)]:
                if buf.done:
                    continue
                try:
                    item = q.get(timeout=0.05)
                    if item is None:
                        buf.finalize()
                    else:
                        buf.feed(item)
                    break
                except queue.Empty:
                    pass

    # flush whatever's left
    drain(buf_a, q_a)
    drain(buf_b, q_b)
    print_rows()
    print()


def main() -> None:
    print("mosaicAI parallel generation demo  (Ctrl-C or 'quit' to exit)")
    print("Submit two prompts at once and watch both stream side by side.\n")

    while True:
        try:
            registered = get_registered_adapters()
        except requests.exceptions.ConnectionError:
            print("[ERROR] Cannot reach middleware at", MIDDLEWARE_URL)
            sys.exit(1)

        active = get_active_adapters()
        print_adapter_table(registered, active)

        aid_a, name_a = _pick_adapter(registered, "Prompt A")
        query_a = input(f"\n  [Prompt A / {name_a}] Message: ").strip()
        if query_a.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            sys.exit(0)
        if not query_a:
            continue

        print()

        aid_b, name_b = _pick_adapter(registered, "Prompt B")
        query_b = input(f"\n  [Prompt B / {name_b}] Message: ").strip()
        if query_b.lower() in ("quit", "exit", "q"):
            print("Bye.")
            sys.exit(0)
        if not query_b:
            continue

        term_w = shutil.get_terminal_size((120, 40)).columns
        col_w = (term_w - len(SEP)) // 2
        rule = "─" * col_w

        print()
        print(f"  {'[A] ' + name_a:<{col_w}}{SEP}[B] {name_b}")
        print(f"  {rule}{SEP}{rule}")

        _render_parallel(query_a, aid_a, query_b, aid_b, col_w)

        print(f"  {rule}{SEP}{rule}\n")

        again = input("  Run another pair? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye.")
            sys.exit(0)
        print()


if __name__ == "__main__":
    main()