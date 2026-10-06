#!/usr/bin/env python3
import gc
import sys
from array import array


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


def _myers_core(A: list, B: list, max_d: int | None = None) -> list[tuple[str, any]] | None:
    """Core O(ND) Myers diff algorithm using 0-indexed diagonal arrays.

    Returns None if D exceeds max_d.
    """
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

    x, y = 0, 0
    while x < N and y < M and id_A[x] == id_B[y]:
        x += 1
        y += 1

    if x >= N and y >= M:
        return [(" ", item) for item in A]

    # Each V is saved as a 4-byte C int array for the backtrack. A list of
    # Python ints costs ~36 bytes per entry, and history holds ~D*D/2 entries.
    history = [array("i", [x])]
    prev = [x]
    d = 0
    found = False
    while not found:
        d += 1
        if max_d is not None and d > max_d:
            return None
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
                break

        history.append(array("i", curr))
        prev = curr

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


# Myers costs about D*D/2 Python steps, which is too slow once D reaches a few
# thousand. Past MYERS_MAX_D we switch to bit-parallel LCS if its table fits.
# ponytail: the table keeps one bit row per item of the longer side, so huge
# dense diffs that don't fit still run on Myers; a Hirschberg-style split
# would make it linear space if that's ever needed.
MYERS_MAX_D = 1000
BITS_MAX_BYTES = 150 * 2**20


def diff_sequences(A: list, B: list) -> list[tuple[str, any]]:
    """Minimal edit script: Myers first, bit-parallel LCS when D gets large."""
    short, long_ = sorted((len(A), len(B)))
    fits = (long_ + 1) * (short // 8 + 40) <= BITS_MAX_BYTES
    script = _myers_core(A, B, MYERS_MAX_D if fits else None)
    return script if script is not None else _bit_lcs(A, B)


def _bit_lcs(A: list, B: list) -> list[tuple[str, any]]:
    """Minimal edit script via bit-parallel LCS (Allison-Dix / Hyyro).

    Bit i of rows[j] is 0 exactly when LCS(A[:i+1], B[:j]) = LCS(A[:i], B[:j]) + 1,
    so each row is one DP column packed into a Python int.
    """
    if len(A) > len(B):  # keep the bit rows short
        flip = {" ": " ", "-": "+", "+": "-"}
        return [(flip[op], item) for op, item in _bit_lcs(B, A)]

    n = len(A)
    match = {}
    for i, item in enumerate(A):
        match[item] = match.get(item, 0) | (1 << i)
    full = (1 << n) - 1
    v = full
    rows = [v]
    for item in B:
        u = v & match.get(item, 0)
        v = ((v + u) | (v - u)) & full
        rows.append(v)

    # Backtrack from (n, len(B)). Equal items are always safe to keep.
    script = []
    i, j = n, len(B)
    while i and j:
        if A[i - 1] == B[j - 1]:
            script.append((" ", A[i - 1]))
            i -= 1
            j -= 1
        elif rows[j] >> (i - 1) & 1:  # LCS unchanged without A[i-1]
            script.append(("-", A[i - 1]))
            i -= 1
        else:
            script.append(("+", B[j - 1]))
            j -= 1
    script.extend(("-", A[k]) for k in range(i - 1, -1, -1))
    script.extend(("+", B[k]) for k in range(j - 1, -1, -1))
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
    char_script = prefix + diff_sequences(a[start:end_a], b[start:end_b]) + suffix

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
        # A line that appears in only one file can never be kept, so Myers runs
        # on the other lines only. The minimum stays the same and D shrinks.
        in_a, in_b = set(mid_a), set(mid_b)
        ka = [i for i, line in enumerate(mid_a) if line in in_b]
        kb = [j for j, line in enumerate(mid_b) if line in in_a]
        mid_diff = diff_sequences([mid_a[i] for i in ka], [mid_b[j] for j in kb])

        def flush_block(minuses, pluses):
            for line in minuses:
                out.write(b"-" + line + b"\n")
            num_pairs = min(len(minuses), len(pluses))
            for i, line in enumerate(pluses):
                out.write(b"+" + line + b"\n")
                if command == "highlight" and i < num_pairs:
                    hl = compute_highlight(minuses[i], line)
                    out.write(hl.encode("utf-8") + b"\n")

        # p, q walk the filtered lists; ka[p], kb[q] map a keep line back to
        # mid_a / mid_b. Everything between two keep lines is one change block.
        a_ptr = b_ptr = p = q = 0
        for op, _ in mid_diff:
            if op == " ":
                i, j = ka[p], kb[q]
                if i > a_ptr or j > b_ptr:
                    flush_block(mid_a[a_ptr:i], mid_b[b_ptr:j])
                out.write(b" " + mid_a[i] + b"\n")
                a_ptr, b_ptr = i + 1, j + 1
                p += 1
                q += 1
            elif op == "-":
                p += 1
            else:
                q += 1
        flush_block(mid_a[a_ptr:], mid_b[b_ptr:])

    # 3. Stream common suffix directly
    for i in range(end_a, len(lines_a)):
        out.write(b" " + lines_a[i] + b"\n")


def main() -> int:
    # Nothing here creates reference cycles, and the cycle collector otherwise
    # rescans the ~1M script tuples again and again.
    gc.disable()
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