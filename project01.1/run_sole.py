from core.classes import SoleExp
from core.utils import load_yaml
import fire


def search_sole(path):
    cfg = load_yaml(path)
    sexp = SoleExp(cfg)
    sexp.run()
    df = sexp.save_csv()
    return df

if __name__ == "__main__":
    fire.Fire(search_sole)