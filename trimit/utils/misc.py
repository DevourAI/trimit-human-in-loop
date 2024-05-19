def lazy_method_decorator(lazy_method):
    def decorator(method):
        def wrapper(*args, **kwargs):
            # Call the lazy method first
            lazy_method()
            # Then call the original method
            return method(*args, **kwargs)

        return wrapper

    return decorator


def union_list_of_intervals(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    # Combine the two lists
    intervals = left + right

    # Sort the intervals based on the starting times
    intervals.sort(key=lambda x: x[0])

    # The list to store the merged intervals
    merged = []

    for start, end in intervals:
        # If the list is empty or there's no overlap, add the interval
        if not merged or merged[-1][1] < start:
            merged.append((start, end))
        else:
            # There is an overlap or adjacency, so merge the intervals
            # Update the end of the last interval in the list if necessary
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    return merged
