import fire
import yaml

from core.classes import Speculator
from core.dataclass import Config, Parameters, Functions



def run_genetic_algorithm(config_path):
    with open(config_path,"r") as config_file:
        config_data = yaml.safe_load(config_file)
    cfg = Config(
        name=config_data["name"],
        path=config_data["path"],
        file_name= config_data["file_name"],
        num_trial=config_data["num_trial"],
        parameters=Parameters(**config_data["parameters"]),
        functions=Functions(**config_data["functions"])
    )
    spc = Speculator(cfg)
    spc.run()
    df = spc.save_csv()
    return df
    

if __name__ == "__main__":
    fire.Fire(run_genetic_algorithm)