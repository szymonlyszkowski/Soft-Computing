INPUT = 0
DESIRED_OUTPUT = 1


def teaching_patterns_with_desired_outputs():
    patterns = []
    patterns.append(([1, 0, 0, 0], [1, 0, 0, 0]))
    patterns.append(([0, 1, 0, 0], [0, 1, 0, 0]))
    patterns.append(([0, 0, 1, 0], [0, 0, 1, 0]))
    patterns.append(([0, 0, 0, 1], [0, 0, 0, 1]))
    return patterns
