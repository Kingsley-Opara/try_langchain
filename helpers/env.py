from decouple import config, Config, RepositoryEnv
import pathlib
from functools import lru_cache

BASE_DIR = pathlib.Path(__file__).parent.resolve()

REPO_DIR = BASE_DIR.parent

BASE_DIR_ENV = BASE_DIR/'.env'

REPO_DIR_ENV = REPO_DIR/".env"


def get_config():
    if BASE_DIR_ENV.exists():
        return Config(RepositoryEnv(str(BASE_DIR_ENV)))
    
    if REPO_DIR_ENV.exists():
        return Config(RepositoryEnv(str(REPO_DIR_ENV)))
    
    return config

config = get_config()