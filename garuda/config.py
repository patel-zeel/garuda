import os
from beartype import beartype
from beartype.typing import Literal
from garuda.base import logger

def enable_tqdm():
    os.environ["GARUDA_DISABLE_TQDM"] = "False"
    logger.info("TQDM progress bar enabled")
    
def disable_tqdm():
    os.environ["GARUDA_DISABLE_TQDM"] = "True"
    logger.info("TQDM progress bar disabled")

@beartype
def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
    logger.setLevel(level)
    print(f"Log level set to {level}")
    
@beartype
def set_n_cpus(n: int):
    os.environ["GARUDA_N_CPUS"] = str(n)
    logger.info(f"Number of CPUs set to {n}")
    
@beartype
def get_n_cpus() -> int:
    return int(os.environ.get("GARUDA_N_CPUS", 1))