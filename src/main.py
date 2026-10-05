import sys


def read_file_lines(path: str) -> list[bytes]:
    """Reads a file as raw bytes, splits on b'\\n', drops trailing empty piece,

    and preserves any carriage return b'\\r'.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) == 0:
        return []

    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def myers_diff(a: list, b: list) -> list[tuple[str, any]]:
    """Computes a minimal edit script between sequences a and b using Myers'

    algorithm.

    Includes prefix/suffix trimming to run in O(D) time.
    """
    n, m = len(a), len(b)

    if a == b:
        return [(" ", item) for item in a]

    # 1. Common prefix trimming
    start = 0
    while start < n and start < m and a[start] == b[start]:
        start += 1

    # 2. Common suffix trimming
    end_a, end_b = n, m
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1

    prefix = [(" ", a[i]) for i in range(start)]
    suffix = [(" ", a[i]) for i in range(end_a, n)]

    mid_a = a[start:end_a]
    mid_b = b[start:end_b]

    mid_script = myers_core(mid_a, mid_b)
    return prefix + mid_script + suffix


def myers_core(a: list, b: list) -> list[tuple[str, any]]:
    """Core Myers O(ND) algorithm for the trimmed middle section."""
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [("+", item) for item in b]
    if m == 0:
        return [("-", item) for item in a]

    max_d = n + m
    offset = max_d
    v = [0] * (2 * max_d + 1)
    trace = []

    for d in range(max_d + 1):
        for k in range(-d, d + 1, 2):
            idx = k + offset
            if k == -d or (k != d and v[idx - 1] < v[idx + 1]):
                x = v[idx + 1]  # vertical step (insert from b)
            else:
                x = v[idx - 1] + 1  # horizontal step (delete from a)
            y = x - k

            # Snake: follow matching elements
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1

            v[idx] = x
            if x >= n and y >= m:
                trace.append(v[offset - d : offset + d + 1])
                return backtrack(trace, a, b, d, k)

        trace.append(v[offset - d : offset + d + 1])

    return []


def backtrack(
    trace: list[list[int]], a: list, b: list, d: int, k: int
) -> list[tuple[str, any]]:
    """Backtracks through trace to reconstruct the shortest edit script."""
    n, m = len(a), len(b)
    script = []
    x, y = n, m

    for step in range(d, 0, -1):
        v_prev = trace[step - 1]
        idx_prev = step - 1

        def get_v_prev(k_val: int) -> int:
            return v_prev[k_val + idx_prev]

        if k == -step:
            prev_k = k + 1
        elif k == step:
            prev_k = k - 1
        elif get_v_prev(k - 1) < get_v_prev(k + 1):
            prev_k = k + 1
        else:
            prev_k = k - 1

        prev_x = get_v_prev(prev_k)
        prev_y = prev_x - prev_k

        if prev_k == k + 1:
            x_edit = prev_x
            y_edit = prev_y + 1
        else:
            x_edit = prev_x + 1
            y_edit = prev_y

        while x > x_edit and y > y_edit:
            script.append((" ", a[x - 1]))
            x -= 1
            y -= 1

        if prev_k == k + 1:
            script.append(("+", b[y_edit - 1]))
        else:
            script.append(("-", a[x_edit - 1]))

        x, y, k = prev_x, prev_y, prev_k

    while x > 0 and y > 0:
        script.append((" ", a[x - 1]))
        x -= 1
        y -= 1

    script.reverse()
    return script


def indices_to_ranges(indices: list[int]) -> str:
    """Converts character indices to formatted, merged half-open ranges."""
    if not indices:
        return "."
    ranges = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev + 1))
            start = idx
            prev = idx
    ranges.append((start, prev + 1))
    return ",".join(f"{s}-{e}" for s, e in ranges)


def compute_highlight(old_bytes: bytes, new_bytes: bytes) -> str:
    """Computes character differences for paired lines in Unicode code

    points.
    """
    old_str = old_bytes.decode("utf-8")
    new_str = new_bytes.decode("utf-8")

    chars_old = list(old_str)
    chars_new = list(new_str)

    char_script = myers_diff(chars_old, chars_new)

    old_idx = 0
    new_idx = 0
    del_indices = []
    ins_indices = []

    for op, _ in char_script:
        if op == " ":
            old_idx += 1
            new_idx += 1
        elif op == "-":
            del_indices.append(old_idx)
            old_idx += 1
        elif op == "+":
            ins_indices.append(new_idx)
            new_idx += 1

    range_old = indices_to_ranges(del_indices)
    range_new = indices_to_ranges(ins_indices)
    return f"? {range_old} | {range_new}"


def run_diff(command: str, lines_a: list[bytes], lines_b: list[bytes]) -> None:
    """Executes diff and outputs formatted lines obeying the delete-first

    rule.
    """
    out = sys.stdout.buffer

    if not lines_a and not lines_b:
        return

    # Intern lines to integer IDs for fast comparison
    intern_map: dict[bytes, int] = {}
    tokens_a = [intern_map.setdefault(l, len(intern_map)) for l in lines_a]
    tokens_b = [intern_map.setdefault(l, len(intern_map)) for l in lines_b]

    diff_tokens = myers_diff(tokens_a, tokens_b)

    a_ptr, b_ptr = 0, 0
    script = []
    for op, _ in diff_tokens:
        if op == " ":
            script.append((" ", lines_a[a_ptr]))
            a_ptr += 1
            b_ptr += 1
        elif op == "-":
            script.append(("-", lines_a[a_ptr]))
            a_ptr += 1
        elif op == "+":
            script.append(("+", lines_b[b_ptr]))
            b_ptr += 1

    current_minuses: list[bytes] = []
    current_pluses: list[bytes] = []

    def flush_block():
        nonlocal current_minuses, current_pluses
        if not current_minuses and not current_pluses:
            return

        if command == "lines":
            for line in current_minuses:
                out.write(b"-" + line + b"\n")
            for line in current_pluses:
                out.write(b"+" + line + b"\n")
        else:
            for line in current_minuses:
                out.write(b"-" + line + b"\n")

            num_pairs = min(len(current_minuses), len(current_pluses))
            for i, line in enumerate(current_pluses):
                out.write(b"+" + line + b"\n")
                if i < num_pairs:
                    h_str = compute_highlight(current_minuses[i], line)
                    out.write(h_str.encode("utf-8") + b"\n")

        current_minuses = []
        current_pluses = []

    for op, line in script:
        if op == " ":
            flush_block()
            out.write(b" " + line + b"\n")
        elif op == "-":
            current_minuses.append(line)
        elif op == "+":
            current_pluses.append(line)

    flush_block()


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