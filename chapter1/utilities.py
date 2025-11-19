def integrate_trace(trace: list[tuple[float, int | float]], initial_value: float = 0) -> float:
    """
    Integrate a trace from a simulation run

    `trace` is a list of (time, value) pairs, assumed to be in chronological order.

    `initial_value` is the value of the trace at time 0.
    """
    result: float = 0
    x_prev: float = 0
    y_prev: float = initial_value

    for x, y in trace:
        result += (x - x_prev) * y_prev
        x_prev = x
        y_prev = y

    return result

# Used for testing


def main():
    test_trace = [(0., 0), (1, 1), (2, 2), (3, 1), (4, 0)]
    test_trace_2 = [(0., 0), (1, 1), (2, 2), (3, 1), (4, 4)]

    assert integrate_trace(test_trace) == 4
    assert integrate_trace(test_trace_2) == 4

    print("Tests pass")


if __name__ == '__main__':
    main()
