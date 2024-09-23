"""Setup file."""

from os import path

from setuptools import find_packages, setup

# Do not edit these constants. They will be updated automatically
# by CI/CD
version_file = path.join(path.dirname(__file__), "zav", "agents_sdk", "version.py")
with open(version_file) as vf:
    exec(vf.read())

HERE = path.abspath(path.dirname(__file__))
README_FILE = path.join(HERE, "README.md")

with open(README_FILE, encoding="utf-8") as fp:
    README = fp.read()

extras_require = {
	"anthropic": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anthropic==0.34.2",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"filelock==3.16.1",
		"frozenlist==1.4.1",
		"fsspec==2024.9.0",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"huggingface-hub==0.25.0",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"tokenizers==0.20.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"bedrock": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anthropic[bedrock]==0.34.2",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"filelock==3.16.1",
		"frozenlist==1.4.1",
		"fsspec==2024.9.0",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"huggingface-hub==0.25.0",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"tokenizers==0.20.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"langchain-anthropic": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anthropic==0.34.2",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"defusedxml==0.7.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"filelock==3.16.1",
		"frozenlist==1.4.1",
		"fsspec==2024.9.0",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"greenlet==3.1.0",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"huggingface-hub==0.25.0",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonpatch==1.33",
		"jsonpointer==3.0.0",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"langchain==0.2.16",
		"langchain-anthropic==0.1.23",
		"langchain-core==0.2.41",
		"langchain-text-splitters==0.2.4",
		"langsmith==0.1.124",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"orjson==3.10.7",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sqlalchemy==2.0.35",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"tokenizers==0.20.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"langchain-aws": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.34.162",
		"botocore==1.34.162",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"greenlet==3.1.0",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonpatch==1.33",
		"jsonpointer==3.0.0",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"langchain==0.2.16",
		"langchain-aws==0.1.18",
		"langchain-core==0.2.41",
		"langchain-text-splitters==0.2.4",
		"langsmith==0.1.124",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"orjson==3.10.7",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sqlalchemy==2.0.35",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"langchain-openai": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"greenlet==3.1.0",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonpatch==1.33",
		"jsonpointer==3.0.0",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"langchain==0.2.16",
		"langchain-core==0.2.41",
		"langchain-openai==0.1.25",
		"langchain-text-splitters==0.2.4",
		"langsmith==0.1.124",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"orjson==3.10.7",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"regex==2024.9.11",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sqlalchemy==2.0.35",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"tiktoken==0.7.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"langchain": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.4.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.17.1",
		"azure-storage-blob==12.22.0",
		"blinker==1.8.2",
		"boto3==1.34.162",
		"botocore==1.34.162",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.114.2",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"greenlet==3.1.0",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.9",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonpatch==1.33",
		"jsonpointer==3.0.0",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"langchain==0.2.16",
		"langchain-core==0.2.40",
		"langchain-text-splitters==0.2.4",
		"langsmith==0.1.120",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.0",
		"numpy==1.24.4",
		"openai==1.45.0",
		"orjson==3.10.7",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.1",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"pyyaml==6.0.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sqlalchemy==2.0.34",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"langfuse": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"backoff==2.2.1",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"langfuse==2.50.2",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"wrapt==1.16.0",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
	"openai": [
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.5.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.18.0",
		"azure-storage-blob==12.23.0",
		"blinker==1.8.2",
		"boto3==1.35.23",
		"botocore==1.35.23",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.115.0",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.10",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.1",
		"numpy==1.24.4",
		"openai==1.46.1",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.2",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	]
}

extras_require["all"] = list({dep for deps in extras_require.values() for dep in deps})

setup(
    name="zetaalpha.rag-agents",
    version=__version__,
    description="The Agents SDK is designed to provide a flexible, scalable, and efficient framework for building, testing, and deploying LLM agents.",
    author_email="support@zeta-alpha.com",
    author="Zeta Alpha Vector",
    url="https://github.com/zetaalphavector/platform/",
	install_requires=[
		"aiofiles==24.1.0",
		"aiohappyeyeballs==2.4.0",
		"aiohttp==3.10.5",
		"aiosignal==1.3.1",
		"altair==5.4.1",
		"anyio==4.4.0",
		"async-timeout==4.0.3",
		"attrs==24.2.0",
		"azure-core==1.31.0",
		"azure-identity==1.17.1",
		"azure-storage-blob==12.22.0",
		"blinker==1.8.2",
		"boto3==1.34.162",
		"botocore==1.34.162",
		"cachetools==5.5.0",
		"certifi==2024.8.30",
		"cffi==1.17.1",
		"charset-normalizer==3.3.2",
		"click==8.1.7",
		"cryptography==43.0.1",
		"distro==1.9.0",
		"exceptiongroup==1.2.2",
		"fastapi==0.114.2",
		"frozenlist==1.4.1",
		"gitdb==4.0.11",
		"gitpython==3.1.43",
		"h11==0.14.0",
		"httpcore==1.0.5",
		"httpx==0.27.2",
		"idna==3.9",
		"importlib-resources==6.4.5",
		"isodate==0.6.1",
		"jinja2==3.1.4",
		"jiter==0.5.0",
		"jmespath==1.0.1",
		"jsonschema==4.23.0",
		"jsonschema-specifications==2023.12.1",
		"markdown-it-py==3.0.0",
		"markupsafe==2.1.5",
		"mdurl==0.1.2",
		"msal==1.31.0",
		"msal-extensions==1.2.0",
		"multidict==6.1.0",
		"narwhals==1.8.0",
		"numpy==1.24.4",
		"openai==1.45.0",
		"packaging==24.1",
		"pandas==2.0.3",
		"pillow==10.4.0",
		"pkgutil-resolve-name==1.3.10",
		"portalocker==2.10.1",
		"protobuf==5.28.1",
		"pyarrow==17.0.0",
		"pycparser==2.22",
		"pydantic==1.10.18",
		"pydeck==0.9.1",
		"pygments==2.18.0",
		"pyjwt[crypto]==2.9.0",
		"python-dateutil==2.9.0.post0",
		"python-json-logger==2.0.7",
		"pytz==2024.2",
		"ragelo==0.1.6",
		"referencing==0.35.1",
		"requests==2.32.3",
		"rich==13.8.1",
		"rpds-py==0.20.0",
		"s3transfer==0.10.2",
		"shellingham==1.5.4",
		"six==1.16.0",
		"smmap==5.0.1",
		"sniffio==1.3.1",
		"sse-starlette==2.1.3",
		"starlette==0.38.5",
		"streamlit==1.38.0",
		"tenacity==8.5.0",
		"toml==0.10.2",
		"tornado==6.4.1",
		"tqdm==4.66.5",
		"typer==0.12.5",
		"typing-extensions==4.12.2",
		"tzdata==2024.1",
		"urllib3==1.26.20",
		"uvicorn==0.30.6",
		"watchdog==4.0.2",
		"yarl==1.11.1",
		"zipp==3.20.2",
	],
    extras_require=extras_require,
    packages=find_packages(exclude=("tests*",)),
    include_package_data=True,
    long_description=README,
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Utilities",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "console_scripts": [
            "rag_agents=zav.agents_sdk.cli.main:app",
        ]
    },
)
