import json


def load_api_key(file_path: str) -> str:
    """
    Retrieves the API key from the given file.
    :param file_path: file path to key file
    :return: API key
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["secret_key"]


def read_json_file(file_path: str) -> dict | list | None:
    """
    Reads a JSON file from the given path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data (dictionary or list depending on the JSON structure).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None
