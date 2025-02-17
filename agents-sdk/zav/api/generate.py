import argparse
import os
from importlib import import_module

import yaml
from fastapi import FastAPI

from zav.api.setup_routers import setup_routers

if __name__ == "__main__":
    app = FastAPI()
    app.openapi_version = "3.0.2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=str,
        default="./src/openapi/spec.yaml",
        help="Where to store the generated OpenAPI spec.",
    )

    parser.add_argument(
        "--routers-module",
        type=str,
        default="src.controllers",
        help="Module where `routers` are defined.",
    )
    args = parser.parse_args()

    controllers = import_module(args.routers_module)
    setup_routers(app=app, routers=controllers.routers)

    os.makedirs(os.path.dirname(args.destination), exist_ok=True)
    with open(args.destination, "w") as f:
        yaml.dump(app.openapi(), f)
