from typing import Any
from fire import Fire
from version2 import load_metrics
import yaml

file_paths = ["D:\USERS\Exeter\Courses\Module ECMM409 (2023) Nature-Inspired Computation\Nature_inspired\project01\dataset\burma14.xml",
        "D:\USERS\Exeter\Courses\Module ECMM409 (2023) Nature-Inspired Computation\Nature_inspired\project01\dataset\brazil58.xml"]

def hello(text):
    return f"Hello {text}"

class Genetic4TSP:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
## TODO:1.write a yaml loader function to distribute the parameter into the classes/functions
## 2. make this code run twice so we dont have to run it twice outside
## 3. at each run, write the infomation file in json object
def run(yaml):
    distance_metrics = [load_metrics(path) for path in file_paths]
    for _ in range(len(distance_metrics)):
        pass
    

if __name__ == "__main__":
    Fire(hello)
