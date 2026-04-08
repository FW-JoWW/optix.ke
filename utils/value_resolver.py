import difflib

def resolve_value(value, possible_values):
    """
    Maps user input value to closest dataset value
    """
    value = str(value).lower().strip()
    choices = [str(v).lower() for v in possible_values]

    match = difflib.get_close_matches(value, choices, n=1, cutoff=0.6)

    if match:
        # return original casing from dataset
        for v in possible_values:
            if str(v).lower() == match[0]:
                return v

    return value