from typing import Dict
from pydantic import BaseModel
import shutil
import importlib
import os


class ConfigSchema(BaseModel):
    learning_rate: float
    momentum: float
    batch_size: int
    num_epochs: int


DEFAULT_CONFIG_FILE='configs/default.py'
    
def get_config(config_file=DEFAULT_CONFIG_FILE):

    module_name = "temp_config"
    shutil.copyfile(config_file, os.path.join(f"{module_name}.py"))
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    config = module.get_config()

    config = ConfigSchema(**config)

    return config

if __name__ == "__main__":


    config = get_config()
    print(config)
