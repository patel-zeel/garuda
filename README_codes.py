def config_tqdm():
    from garuda.config import enable_tqdm, disable_tqdm
    
    enable_tqdm()  # Enable tqdm progress bar across the library
    disable_tqdm()  # Disable tqdm progress bar across the library    
    
def config_log_level():
    from garuda.config import set_log_level
    from garuda.base import logger
    
    def _log_everything():
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    # default log level is INFO
    set_log_level("DEBUG")
    _log_everything()
    
    set_log_level("INFO")
    _log_everything()
    
    set_log_level("WARNING")
    _log_everything()
    
    set_log_level("ERROR")
    _log_everything()
    
    set_log_level("CRITICAL")
    _log_everything()