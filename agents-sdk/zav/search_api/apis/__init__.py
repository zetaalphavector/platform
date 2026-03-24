
# flake8: noqa

# Import all APIs into this package.
# If you have many APIs here with many many models used in each API this may
# raise a `RecursionError`.
# In order to avoid this, import only the API that you directly need like:
#
#   from .api.answers_api import AnswersApi
#
# or import this package, but before doing it, use:
#
#   import sys
#   sys.setrecursionlimit(n)

# Import APIs into API package:
from zav.search_api.api.document_assets_api import DocumentAssetsApi
from zav.search_api.api.documents_api import DocumentsApi
from zav.search_api.api.sources_api import SourcesApi
from zav.search_api.api.tenant_settings_api import TenantSettingsApi
