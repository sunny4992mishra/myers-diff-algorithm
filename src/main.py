#!/usr/bin/env python3
import sys
from array import array


def read_lines(path):
    """Read a file as raw bytes and split it into lines on b"\\n".

    A final newline does not create an extra empty line, and any b"\\r" stays
    part of its line.
    """
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


def myers(a, b):
    """Myers' O(ND) diff. Returns the (i, j) pairs with a[i] == b[j] that a
    shortest edit script keeps, in order.

    v[k + off] is the furthest x reached on diagonal k = x - y.
    """
    n, m = len(a), len(b)
    off = n + m + 1
    v = [0] * (2 * off + 1)
    trace = []  # trace[d][k + d] = v[k] before step d, for the backtrack

    for d in range(n + m + 1):
        # 4-byte ints instead of a list keeps the trace small.
        trace.append(array("i", v[off - d : off + d + 1]))
        for k in range(-d, d + 1, 2):
            # Come down from diagonal k+1 (insert) or right from k-1 (delete),
            # whichever got further.
            if k == -d or (k != d and v[off + k - 1] < v[off + k + 1]):
                x = v[off + k + 1]
            else:
                x = v[off + k - 1] + 1
            y = x - k
            # Follow the snake: equal items are free diagonal moves.
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1
            v[off + k] = x
            if x >= n and y >= m:
                return backtrack(trace, n, m)


def backtrack(trace, n, m):
    """Walk back from (n, m) and collect the diagonal moves (kept pairs)."""
    pairs = []
    x, y = n, m
    for d in range(len(trace) - 1, 0, -1):
        v = trace[d]
        k = x - y
        # Same choice as the forward pass, using v from before step d.
        if k == -d or (k != d and v[k - 1 + d] < v[k + 1 + d]):
            prev_k = k + 1
        else:
            prev_k = k - 1
        prev_x = v[prev_k + d]
        prev_y = prev_x - prev_k
        while x > prev_x and y > prev_y:  # back along the snake
            x -= 1
            y -= 1
            pairs.append((x, y))
        x, y = prev_x, prev_y
    while x > 0 and y > 0:  # the snake from (0, 0) at d = 0
        x -= 1
        y -= 1
        pairs.append((x, y))
    pairs.reverse()
    return pairs


def matching_pairs(a, b):
    """The (i, j) pairs kept by a shortest edit script of a and b.

    Before running Myers, two shortcuts that never change the minimum:
    equal items at the start and end are always kept, and items that never
    appear on the other side can never be kept.
    """
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1

    in_a, in_b = set(a[start:end_a]), set(b[start:end_b])
    idx_a = [i for i in range(start, end_a) if a[i] in in_b]
    idx_b = [j for j in range(start, end_b) if b[j] in in_a]
    middle = myers([a[i] for i in idx_a], [b[j] for j in idx_b])

    pairs = [(i, i) for i in range(start)]
    pairs += [(idx_a[x], idx_b[y]) for x, y in middle]
    pairs += [(end_a + t, end_b + t) for t in range(len(a) - end_a)]
    return pairs


def to_ranges(positions):
    """[3, 4, 5, 9] -> "3-6,9-10". Ranges are start-end with end excluded."""
    if not positions:
        return "."
    ranges = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p != prev + 1:
            ranges.append(f"{start}-{prev + 1}")
            start = p
        prev = p
    ranges.append(f"{start}-{prev + 1}")
    return ",".join(ranges)


def highlight(old, new):
    """The "? old | new" line: the characters that are not kept."""
    a, b = old.decode("utf-8"), new.decode("utf-8")  # str counts code points
    pairs = matching_pairs(a, b)
    kept_a = {i for i, _ in pairs}
    kept_b = {j for _, j in pairs}
    changed_a = [i for i in range(len(a)) if i not in kept_a]
    changed_b = [j for j in range(len(b)) if j not in kept_b]
    return f"? {to_ranges(changed_a)} | {to_ranges(changed_b)}".encode()


def print_diff(command, a, b, out):
    """Everything between two kept lines is one change block: print its
    deleted lines first, then its inserted lines."""
    a_pos = b_pos = 0
    # The extra pair (len(a), len(b)) flushes the last block.
    for i, j in matching_pairs(a, b) + [(len(a), len(b))]:
        if i > a_pos or j > b_pos:
            deleted, inserted = a[a_pos:i], b[b_pos:j]
            for line in deleted:
                out.write(b"-" + line + b"\n")
            for t, line in enumerate(inserted):
                out.write(b"+" + line + b"\n")
                # Pair the t-th inserted line with the t-th deleted line.
                if command == "highlight" and t < len(deleted):
                    out.write(highlight(deleted[t], line) + b"\n")
        if i < len(a):
            out.write(b" " + a[i] + b"\n")
        a_pos, b_pos = i + 1, j + 1


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("lines", "highlight"):
        print("usage: main.py lines|highlight A B", file=sys.stderr)
        return 2
    command, path_a, path_b = sys.argv[1:]
    # Read both files before printing anything.
    try:
        a = read_lines(path_a)
        b = read_lines(path_b)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print_diff(command, a, b, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
