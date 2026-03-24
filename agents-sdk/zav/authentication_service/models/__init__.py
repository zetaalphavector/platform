# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from zav.authentication_service.model.api_key_string import ApiKeyString
from zav.authentication_service.model.card_style_string import CardStyleString
from zav.authentication_service.model.email_string import EmailString
from zav.authentication_service.model.generic_error import GenericError
from zav.authentication_service.model.homepage_string import HomepageString
from zav.authentication_service.model.index_cluster import IndexCluster
from zav.authentication_service.model.index_clusters import IndexClusters
from zav.authentication_service.model.jwt import JWT
from zav.authentication_service.model.jwt_string import JWTString
from zav.authentication_service.model.jwts import JWTs
from zav.authentication_service.model.password_string import PasswordString
from zav.authentication_service.model.role_db_with_tenant import RoleDBWithTenant
from zav.authentication_service.model.uuid_string import UUIDString
from zav.authentication_service.model.user_changelog import UserChangelog
from zav.authentication_service.model.user_credentials import UserCredentials
from zav.authentication_service.model.user_payload import UserPayload
from zav.authentication_service.model.user_roles import UserRoles
from zav.authentication_service.model.user_settings import UserSettings
from zav.authentication_service.model.user_settings_patch import UserSettingsPatch
from zav.authentication_service.model.user_tenants import UserTenants
from zav.authentication_service.model.verify_form import VerifyForm
