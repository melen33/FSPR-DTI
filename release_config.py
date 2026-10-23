"""Load a public experiment YAML through the existing command-line interface."""

from pathlib import Path

from omegaconf import OmegaConf


def load_release_config(path, parser, explicit_argv):
    config = OmegaConf.load(Path(path))
    sections = {"run", "model", "structure", "training"}
    extra = set(config.keys()) - sections
    if extra:
        raise ValueError("Unknown experiment configuration sections: " + ", ".join(sorted(extra)))
    if "training" not in config or "TRAIN" not in config.training:
        raise ValueError("Experiment configuration requires training.TRAIN")

    options = {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }
    config_argv = []
    for section in ("run", "model", "structure"):
        values = config[section] if section in config else OmegaConf.create({})
        if not OmegaConf.is_config(values):
            raise ValueError(f"{section} must be a mapping")
        for name, value in values.items():
            option = "--" + name
            if option not in options:
                raise ValueError(f"Unknown {section} option: {name}")
            action = options[option]
            if action.nargs == 0:
                if value is not True:
                    raise ValueError(f"Flag {name} must be true; omit it otherwise")
                config_argv.append(option)
            else:
                if value is None:
                    raise ValueError(f"Option {name} cannot be null")
                items = value if isinstance(value, (list, tuple)) else [value]
                config_argv.extend([option, *map(str, items)])
    return config, config_argv + list(explicit_argv)
