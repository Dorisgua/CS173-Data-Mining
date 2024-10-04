import time
def map_to_range(number):
    """
    Maps each input number to its corresponding range.
    """
    range_width = 0.1
    range_start = int(number / range_width) * range_width
    range_end = range_start + range_width
    return f"[{range_start}, {range_end})"


if __name__ == "__main__":
    file_path = "/home/liuwt/scores_all_data.txt"

    start_time = time.time()
    with open(file_path, 'r') as file:
        numbers = file.read().split()
    range_frequency = {}
    for number in numbers:
        number = float(number)
        range_key = map_to_range(number)
        if range_key in range_frequency:
            range_frequency[range_key] += 1
        else:
            range_frequency[range_key] = 1

    end_time = time.time()
    execution_time = end_time - start_time

    for key, value in range_frequency.items():
        print(f"Range {key}: {value}")

    print("Execution Time:", execution_time, "seconds")
