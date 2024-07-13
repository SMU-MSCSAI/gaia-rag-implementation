import os

from gaia_framework.utils.data_object import DataObject

def log_dataobject_step(data_object: DataObject, 
                        step_description: str, 
                        log_file: str = "../../data/data_processing_log.txt"):
    with open(log_file, "a") as f:
        f.write(f"Step: {step_description}\n")
        f.write(f"DataObject State: {data_object.to_json()}\n")
        f.write("\n-----------------------------------\n")

def reset_log_file(log_file: str = "data_processing_log.txt"):
    with open(log_file, "w") as f:
        f.write("Data Processing Log\n")
        f.write("===================================\n\n")
