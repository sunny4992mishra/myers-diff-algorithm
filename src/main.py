#!/usr/bin/env python3
import sys


def read_file_lines(path: str) -> list[bytes]:
    """Reads a file as raw bytes, splits on b'\\n', drops the trailing empty

    piece if present, and preserves any carriage return b'\\r'.
    """
    with open(path, "rb") as f:
        data = f.read()

    if not data:
        return []

    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def trim_ends(a: list, b: list) -> tuple[int, int, int]:
    """Finds matching prefix length and suffix indices in O(N)."""
    n, m = len(a), len(b)
    start = 0
    while start < n and start < m and a[start] == b[start]:
        start += 1

    end_a, end_b = n, m
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1

    return start, end_a, end_b


def _myers_core(A: list, B: list) -> list[tuple[str, any]]:
    """Core O(ND) Myers diff algorithm using 0-indexed diagonal arrays."""
    N, M = len(A), len(B)
    if N == 0 and M == 0:
        return []
    if N == 0:
        return [("+", item) for item in B]
    if M == 0:
        return [("-", item) for item in A]

    item_ids = {}
    id_A = [item_ids.setdefault(item, len(item_ids)) for item in A]
    id_B = [item_ids.setdefault(item, len(item_ids)) for item in B]

    history = []
    x, y = 0, 0
    while x < N and y < M and id_A[x] == id_B[y]:
        x += 1
        y += 1
    history.append([x])

    if x >= N and y >= M:
        return [(" ", item) for item in A]

    d = 0
    found = False
    while not found:
        d += 1
        prev = history[d - 1]
        curr = [0] * (d + 1)

        for i in range(d + 1):
            k = -d + 2 * i
            if i == 0:
                x = prev[0]
            elif i == d:
                x = prev[d - 1] + 1
            else:
                p_left, p_right = prev[i - 1], prev[i]
                x = p_right if p_left < p_right else p_left + 1

            y = x - k
            while x < N and y < M and id_A[x] == id_B[y]:
                x += 1
                y += 1

            curr[i] = x
            if x >= N and y >= M:
                found = True
                history.append(curr)
                break

        if not found:
            history.append(curr)

    # Backtrack shortest edit script
    script = []
    curr_x, curr_y = N, M

    for step_d in range(d, 0, -1):
        k = curr_x - curr_y
        i = (k + step_d) // 2
        prev = history[step_d - 1]

        if i == 0 or (i != step_d and prev[i - 1] < prev[i]):
            prev_k, prev_i = k + 1, i
        else:
            prev_k, prev_i = k - 1, i - 1

        prev_x = prev[prev_i]
        prev_y = prev_x - prev_k

        if prev_k == k - 1:
            snake_len = curr_x - (prev_x + 1)
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", A[prev_x + 1 + s]))
            script.append(("-", A[prev_x]))
        else:
            snake_len = curr_x - prev_x
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", B[prev_y + 1 + s]))
            script.append(("+", B[prev_y]))

        curr_x, curr_y = prev_x, prev_y

    for s in range(curr_x - 1, -1, -1):
        script.append((" ", A[s]))

    script.reverse()
    return script


def indices_to_ranges(indices: list[int]) -> str:
    """Formats 0-indexed character indices into merged half-open ranges [start,

    end).
    """
    if not indices:
        return "."
    ranges = []
    start = indices[0]
    end = start + 1
    for idx in indices[1:]:
        if idx == end:
            end += 1
        else:
            ranges.append(f"{start}-{end}")
            start = idx
            end = idx + 1
    ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def compute_highlight(old_bytes: bytes, new_bytes: bytes) -> str:
    """Computes character differences for paired lines in Unicode code

    points.
    """
    a = list(old_bytes.decode("utf-8"))
    b = list(new_bytes.decode("utf-8"))

    start, end_a, end_b = trim_ends(a, b)
    prefix = [(" ", a[i]) for i in range(start)]
    suffix = [(" ", a[i]) for i in range(end_a, len(a))]
    char_script = prefix + _myers_core(a[start:end_a], b[start:end_b]) + suffix

    old_idx, new_idx = 0, 0
    del_indices, ins_indices = [], []

    for op, _ in char_script:
        if op == "-":
            del_indices.append(old_idx)
            old_idx += 1
        elif op == "+":
            ins_indices.append(new_idx)
            new_idx += 1
        else:
            old_idx += 1
            new_idx += 1

    return f"? {indices_to_ranges(del_indices)} | {indices_to_ranges(ins_indices)}"


def run_diff(command: str, lines_a: list[bytes], lines_b: list[bytes]) -> None:
    """Executes line diff with streaming prefix/suffix and delete-first change

    blocks.
    """
    out = sys.stdout.buffer
    if not lines_a and not lines_b:
        return

    # 1. Stream common prefix directly
    start, end_a, end_b = trim_ends(lines_a, lines_b)
    for i in range(start):
        out.write(b" " + lines_a[i] + b"\n")

    # 2. Diff and flush middle section
    mid_a = lines_a[start:end_a]
    mid_b = lines_b[start:end_b]

    if mid_a or mid_b:
        mid_diff = _myers_core(mid_a, mid_b)
        minuses, pluses = [], []

        def flush_block():
            for line in minuses:
                out.write(b"-" + line + b"\n")
            num_pairs = min(len(minuses), len(pluses))
            for i, line in enumerate(pluses):
                out.write(b"+" + line + b"\n")
                if command == "highlight" and i < num_pairs:
                    hl = compute_highlight(minuses[i], line)
                    out.write(hl.encode("utf-8") + b"\n")
            minuses.clear()
            pluses.clear()

        a_ptr, b_ptr = 0, 0
        for op, _ in mid_diff:
            if op == " ":
                if minuses or pluses:
                    flush_block()
                out.write(b" " + mid_a[a_ptr] + b"\n")
                a_ptr += 1
                b_ptr += 1
            elif op == "-":
                minuses.append(mid_a[a_ptr])
                a_ptr += 1
            elif op == "+":
                pluses.append(mid_b[b_ptr])
                b_ptr += 1

        if minuses or pluses:
            flush_block()

    # 3. Stream common suffix directly
    for i in range(end_a, len(lines_a)):
        out.write(b" " + lines_a[i] + b"\n")


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A_PATH B_PATH", file=sys.stderr)
        return 2

    command, a_path, b_path = sys.argv[1:]

    try:
        lines_a = read_file_lines(a_path)
    except OSError as e:
        print(f"Error reading '{a_path}': {e}", file=sys.stderr)
        return 2

    try:
        lines_b = read_file_lines(b_path)
    except OSError as e:
        print(f"Error reading '{b_path}': {e}", file=sys.stderr)
        return 2

    run_diff(command, lines_a, lines_b)
    return 0


raise SystemExit(main())