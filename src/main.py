import sys


def read_file_lines(path: str) -> list[bytes]:
    """Reads a file as raw bytes, splits on b'\\n', drops the trailing empty

    piece if present, and preserves any carriage return b'\\r'.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) == 0:
        return []

    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def _myers_core(A: list, B: list) -> list[tuple[str, any]]:
    """Core O(ND) Myers diff algorithm using 0-indexed diagonal arrays."""
    N = len(A)
    M = len(B)

    if N == 0 and M == 0:
        return []
    if N == 0:
        return [("+", item) for item in B]
    if M == 0:
        return [("-", item) for item in A]

    # Map elements to unique integer IDs for fast comparison in snakes
    item_ids = {}
    id_A = [item_ids.setdefault(item, len(item_ids)) for item in A]
    id_B = [item_ids.setdefault(item, len(item_ids)) for item in B]

    history = []

    # d = 0: initial snake from (0, 0)
    x = 0
    y = 0
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

        # Diagonals k = -d, -d+2, ..., d mapped to index i in 0 .. d
        for i in range(d + 1):
            k = -d + 2 * i

            if i == 0:
                x = prev[0]  # Vertical move (insertion)
            elif i == d:
                x = prev[d - 1] + 1  # Horizontal move (deletion)
            else:
                p_left = prev[i - 1]  # from k - 1
                p_right = prev[i]  # from k + 1
                x = p_right if p_left < p_right else p_left + 1

            y = x - k

            # Snake: greedily advance along matching diagonals
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

    # Backtrack from (N, M) to (0, 0)
    script = []
    curr_x = N
    curr_y = M

    for step_d in range(d, 0, -1):
        k = curr_x - curr_y
        i = (k + step_d) // 2
        prev = history[step_d - 1]

        if i == 0:
            prev_k = k + 1
            prev_i = 0
        elif i == step_d:
            prev_k = k - 1
            prev_i = step_d - 1
        else:
            if prev[i - 1] < prev[i]:
                prev_k = k + 1
                prev_i = i
            else:
                prev_k = k - 1
                prev_i = i - 1

        prev_x = prev[prev_i]
        prev_y = prev_x - prev_k

        if prev_k == k - 1:
            # Horizontal move: Deletion of A[prev_x]
            snake_len = curr_x - (prev_x + 1)
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", A[prev_x + 1 + s]))
            script.append(("-", A[prev_x]))
        else:
            # Vertical move: Insertion of B[prev_y]
            snake_len = curr_x - prev_x
            for s in range(snake_len - 1, -1, -1):
                script.append((" ", B[prev_y + 1 + s]))
            script.append(("+", B[prev_y]))

        curr_x = prev_x
        curr_y = prev_y

    for s in range(curr_x - 1, -1, -1):
        script.append((" ", A[s]))

    script.reverse()
    return script


def indices_to_ranges(indices: list[int]) -> str:
    """Converts 0-indexed character indices to formatted, merged half-open

    ranges.
    """
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


def myers_diff_chars(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Runs Myers diff on character sequences with prefix/suffix trimming."""
    n, m = len(a), len(b)
    if a == b:
        return [(" ", ch) for ch in a]

    start = 0
    while start < n and start < m and a[start] == b[start]:
        start += 1

    end_a, end_b = n, m
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1

    prefix = [(" ", a[i]) for i in range(start)]
    suffix = [(" ", a[i]) for i in range(end_a, n)]

    mid_script = _myers_core(a[start:end_a], b[start:end_b])
    return prefix + mid_script + suffix


def compute_highlight(old_bytes: bytes, new_bytes: bytes) -> str:
    """Computes character differences for paired lines in Unicode code

    points.
    """
    old_str = old_bytes.decode("utf-8")
    new_str = new_bytes.decode("utf-8")

    chars_old = list(old_str)
    chars_new = list(new_str)

    char_script = myers_diff_chars(chars_old, chars_new)

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
    """Executes diff, streaming common prefix and suffix without intermediate

    tuple allocation, and diffing only the middle segment.
    """
    out = sys.stdout.buffer

    if not lines_a and not lines_b:
        return

    n = len(lines_a)
    m = len(lines_b)

    # 1. Fast path: identical files
    if lines_a == lines_b:
        for line in lines_a:
            out.write(b" " + line + b"\n")
        return

    # 2. Stream common prefix directly
    start = 0
    while start < n and start < m and lines_a[start] == lines_b[start]:
        out.write(b" " + lines_a[start] + b"\n")
        start += 1

    # 3. Find common suffix
    end_a, end_b = n, m
    while (
        end_a > start
        and end_b > start
        and lines_a[end_a - 1] == lines_b[end_b - 1]
    ):
        end_a -= 1
        end_b -= 1

    # 4. Diff only the middle segment
    mid_a = lines_a[start:end_a]
    mid_b = lines_b[start:end_b]

    if mid_a or mid_b:
        mid_diff = _myers_core(mid_a, mid_b)

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

        a_ptr, b_ptr = 0, 0
        for op, _ in mid_diff:
            if op == " ":
                flush_block()
                out.write(b" " + mid_a[a_ptr] + b"\n")
                a_ptr += 1
                b_ptr += 1
            elif op == "-":
                current_minuses.append(mid_a[a_ptr])
                a_ptr += 1
            elif op == "+":
                current_pluses.append(mid_b[b_ptr])
                b_ptr += 1

        flush_block()

    # 5. Stream common suffix directly
    for i in range(end_a, n):
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