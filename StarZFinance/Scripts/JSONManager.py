# Function to parse JSON data and extract LSTM parameters
def parse_lstm_parameters(json_data):
    type_map = {
        "int": int,
        "double": float,
        "string": str,
        "bool": lambda x: x.lower() == "true",
    }

    lstm_params = {}
    for param in json_data.get("LSTM", []):
        name = param["Name"]
        value = param["Value"]
        value_type = param["ValueType"]

        # Convert value to the correct type
        try:
            convert_func = type_map.get(value_type.lower(), str)
            lstm_params[name] = convert_func(value)
        except Exception as e:
            raise ValueError(
                f"Error converting parameter '{name}' with value '{value}': {e}"
            )

    return lstm_params
