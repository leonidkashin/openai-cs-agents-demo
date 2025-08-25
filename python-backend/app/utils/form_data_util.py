from urllib.parse import parse_qs


def form_to_json(query_string):
    # Parse the form-data string into a dictionary
    parsed_data = parse_qs(query_string)

    # Convert the dictionary to a nested structure
    nested_dict = {}
    for key, value in parsed_data.items():
        keys = key.split('[')
        current_dict = nested_dict
        for k in keys[:-1]:
            k = k.strip('[]')
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1].strip('[]')] = value[0]

    return nested_dict
