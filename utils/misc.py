import os, sys

def parse_args_from_config(config_path):

    import importlib.util
    
    spec = importlib.util.spec_from_file_location("get_hpyerparameters", config_path )

    modulevar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulevar)

    return modulevar

