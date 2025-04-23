# Function to parse JSON data and extract model parameters
def parse_model_parameters(json_data, model_name):
    type_map = {
        "int": int,
        "double": float,
        "string": str,
        "bool": lambda x: x.lower() == "true",
    }

    model_params = {}
    for param in json_data.get(model_name, []):
        name = param["Name"]
        value = param["Value"]
        value_type = param["ValueType"]

        # Convert value to the correct type
        try:
            convert_func = type_map.get(value_type.lower(), str)
            model_params[name] = convert_func(value)
        except Exception as e:
            raise ValueError(
                f"Error converting parameter '{name}' with value '{value}': {e}"
            )

    return model_params
