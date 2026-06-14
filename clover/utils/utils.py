import logging

def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    
    logger = logging.getLogger("clover")
    
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    logger.info("Line Magic Set")