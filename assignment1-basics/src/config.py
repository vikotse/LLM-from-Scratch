from . import config_ts, config_owt

def get_default_config(config_name: str = None):
    if not config_name or (config_name != "ts" and config_name != "owt"):
        return config_ts.get_default_config()
    cfg_dict = {"ts": config_ts, "owt": config_owt}
    return cfg_dict[config_name].get_default_config()
