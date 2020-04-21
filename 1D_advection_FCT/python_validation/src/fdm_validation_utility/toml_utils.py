import tomlkit

from fdm_validation_utility import FDM_Problem_Config


def problem_config_from_toml(toml_data):
    return FDM_Problem_Config(
        a=toml_data["Velocity"],
        sigma=toml_data["CFL number"],
        end_time=toml_data["End time"],
    )


def problem_config_to_toml(config, base_toml=None):
    if base_toml is None:
        base_toml = base_toml
    base_toml["Velocity"] = config.a
    base_toml["CFL number"] = config.sigma
    base_toml["End time"] = config.end_time
