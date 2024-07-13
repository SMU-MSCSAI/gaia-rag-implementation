import json

from gaia_framework.utils.data_object import DataObject

def log_dataobject_step(data_object: DataObject, step_description: str, log_file: str = "../../data/data_processing_log.txt"):
    """
    Log the current state of the DataObject with a description of the step.

    Args:
        data_object (DataObject): The DataObject to log.
        step_description (str): A description of the current step in the process.
        log_file (str): The file to write the log to.
    """
    log_entry = {
        "step": step_description,
        "data": data_object.to_dict()
    }
    
    # Truncate the embedding for logging if it exists
    if "embedding" in log_entry["data"] and isinstance(log_entry["data"]["embedding"], list):
        log_entry["data"]["embedding"] = log_entry["data"]["embedding"][:2] + ["..."]

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, indent=4))
        f.write("\n-----------------------------------\n")

def reset_log_file(log_file: str = "data_processing_log.txt"):
    """
    Reset the log file.

    Args:
        log_file (str): The file to reset.
    """
    with open(log_file, "w") as f:
        f.write("Data Processing Log\n")
        f.write("===================================\n\n")
