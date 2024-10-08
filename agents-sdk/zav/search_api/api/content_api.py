"""
    Search

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: latest
    Generated by: https://openapi-generator.tech
"""


import re  # noqa: F401
import sys  # noqa: F401

from zav.search_api.api_client import ApiClient, Endpoint as _Endpoint
from zav.search_api.model_utils import (  # noqa: F401
    check_allowed_values,
    check_validations,
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types
)
from zav.search_api.model.bib_tex import BibTex
from zav.search_api.model.bibtex_id_string import BibtexIDString
from zav.search_api.model.document_list_hit import DocumentListHit
from zav.search_api.model.generic_error import GenericError
from zav.search_api.model.uid_string import UIDString
from zav.search_api.model.uuid_string import UUIDString


class ContentApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client
        self.bibtex_get_endpoint = _Endpoint(
            settings={
                'response_type': (str,),
                'auth': [],
                'endpoint_path': '/entities/content/{uid}/bibtex',
                'operation_id': 'bibtex_get',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'requester_uuid',
                    'uid',
                    'user_roles',
                    'tenant',
                ],
                'required': [
                    'requester_uuid',
                    'uid',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                ]
            },
            root_map={
                'validations': {
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'requester_uuid':
                        (UUIDString,),
                    'uid':
                        (BibtexIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                },
                'attribute_map': {
                    'requester_uuid': 'requester-uuid',
                    'uid': 'uid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                },
                'location_map': {
                    'requester_uuid': 'header',
                    'uid': 'path',
                    'user_roles': 'header',
                    'tenant': 'query',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.bibtexs_get_endpoint = _Endpoint(
            settings={
                'response_type': ([BibTex],),
                'auth': [],
                'endpoint_path': '/entities/contents/bibtex',
                'operation_id': 'bibtexs_get',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'uids',
                    'requester_uuid',
                    'user_roles',
                    'tenant',
                ],
                'required': [
                    'uids',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                    'uids',
                ]
            },
            root_map={
                'validations': {
                    ('uids',): {

                        'min_items': 1,
                    },
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'uids':
                        ([BibtexIDString],),
                    'requester_uuid':
                        (UUIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                },
                'attribute_map': {
                    'uids': 'uids',
                    'requester_uuid': 'requester-uuid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                },
                'location_map': {
                    'uids': 'query',
                    'requester_uuid': 'header',
                    'user_roles': 'header',
                    'tenant': 'query',
                },
                'collection_format_map': {
                    'uids': 'csv',
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )
        self.paper_get_endpoint = _Endpoint(
            settings={
                'response_type': (DocumentListHit,),
                'auth': [],
                'endpoint_path': '/entities/content/{uid}',
                'operation_id': 'paper_get',
                'http_method': 'GET',
                'servers': None,
            },
            params_map={
                'all': [
                    'uid',
                    'user_roles',
                    'tenant',
                    'index_cluster',
                ],
                'required': [
                    'uid',
                ],
                'nullable': [
                ],
                'enum': [
                ],
                'validation': [
                    'index_cluster',
                ]
            },
            root_map={
                'validations': {
                    ('index_cluster',): {

                        'regex': {
                            'pattern': r'^[a-zA-Z0-9-_]+:[a-zA-Z0-9-_]+$',  # noqa: E501
                        },
                    },
                },
                'allowed_values': {
                },
                'openapi_types': {
                    'uid':
                        (UIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                    'index_cluster':
                        (str,),
                },
                'attribute_map': {
                    'uid': 'uid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                    'index_cluster': 'index_cluster',
                },
                'location_map': {
                    'uid': 'path',
                    'user_roles': 'header',
                    'tenant': 'query',
                    'index_cluster': 'query',
                },
                'collection_format_map': {
                }
            },
            headers_map={
                'accept': [
                    'application/json'
                ],
                'content_type': [],
            },
            api_client=api_client
        )

    def bibtex_get(
        self,
        requester_uuid,
        uid,
        **kwargs
    ):
        """Get paper bibtex string  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.bibtex_get(requester_uuid, uid, async_req=True)
        >>> result = thread.get()

        Args:
            requester_uuid (UUIDString):
            uid (BibtexIDString): Limit results to only the given uid.

        Keyword Args:
            user_roles (str): [optional]
            tenant (str): Tenant. [optional] if omitted the server will use the default value of "zetaalpha"
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _spec_property_naming (bool): True if the variable names in the input data
                are serialized names, as specified in the OpenAPI document.
                False if the variable names in the input data
                are pythonic names, e.g. snake case (default)
            _content_type (str/None): force body content-type.
                Default is None and content-type will be predicted by allowed
                content-types and body.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            str
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_spec_property_naming'] = kwargs.get(
            '_spec_property_naming', False
        )
        kwargs['_content_type'] = kwargs.get(
            '_content_type')
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['requester_uuid'] = \
            requester_uuid
        kwargs['uid'] = \
            uid
        return self.bibtex_get_endpoint.call_with_http_info(**kwargs)

    def bibtexs_get(
        self,
        uids,
        **kwargs
    ):
        """Get a list of bibtex strings for given paper uids.  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.bibtexs_get(uids, async_req=True)
        >>> result = thread.get()

        Args:
            uids ([BibtexIDString]): Limit results to only the given document uids.

        Keyword Args:
            requester_uuid (UUIDString): [optional]
            user_roles (str): [optional]
            tenant (str): Tenant. [optional] if omitted the server will use the default value of "zetaalpha"
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _spec_property_naming (bool): True if the variable names in the input data
                are serialized names, as specified in the OpenAPI document.
                False if the variable names in the input data
                are pythonic names, e.g. snake case (default)
            _content_type (str/None): force body content-type.
                Default is None and content-type will be predicted by allowed
                content-types and body.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            [BibTex]
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_spec_property_naming'] = kwargs.get(
            '_spec_property_naming', False
        )
        kwargs['_content_type'] = kwargs.get(
            '_content_type')
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['uids'] = \
            uids
        return self.bibtexs_get_endpoint.call_with_http_info(**kwargs)

    def paper_get(
        self,
        uid,
        **kwargs
    ):
        """Get paper information  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.paper_get(uid, async_req=True)
        >>> result = thread.get()

        Args:
            uid (UIDString): UID of entity or concept you want to retrieve

        Keyword Args:
            user_roles (str): [optional]
            tenant (str): Tenant. [optional] if omitted the server will use the default value of "zetaalpha"
            index_cluster (str): Human friendly name that specifies which index configuration to use during search. The way we convert this string into infrastructure configuration is part of the internal logic of this service. Ideally all possible values should be listed as part of another request along with the description of what each value means.  Currently, the value can be constructed by using knowledge of the index infrastructure. In particular, the value is separated by the `:` character. The part on the left of `:` represents the kubernetes namespace of the index cluster, while the part on the right specifies the infix in the index name.  For example, if a user wants to search in the index cluster located in the `foo` namespace and use the index named `my_tenant_bar_documents`, then the `index_cluster` value should be `foo:bar`. The user may also need to specify the retrieval unit and tenant as part of the request.  > Note: There's currently no endpoint for retrieving all valid values that the `index_cluster` parameter can take. . [optional]
            _return_http_data_only (bool): response data without head status
                code and headers. Default is True.
            _preload_content (bool): if False, the urllib3.HTTPResponse object
                will be returned without reading/decoding response data.
                Default is True.
            _request_timeout (int/float/tuple): timeout setting for this request. If
                one number provided, it will be total request timeout. It can also
                be a pair (tuple) of (connection, read) timeouts.
                Default is None.
            _check_input_type (bool): specifies if type checking
                should be done one the data sent to the server.
                Default is True.
            _check_return_type (bool): specifies if type checking
                should be done one the data received from the server.
                Default is True.
            _spec_property_naming (bool): True if the variable names in the input data
                are serialized names, as specified in the OpenAPI document.
                False if the variable names in the input data
                are pythonic names, e.g. snake case (default)
            _content_type (str/None): force body content-type.
                Default is None and content-type will be predicted by allowed
                content-types and body.
            _host_index (int/None): specifies the index of the server
                that we want to use.
                Default is read from the configuration.
            async_req (bool): execute request asynchronously

        Returns:
            DocumentListHit
                If the method is called asynchronously, returns the request
                thread.
        """
        kwargs['async_req'] = kwargs.get(
            'async_req', False
        )
        kwargs['_return_http_data_only'] = kwargs.get(
            '_return_http_data_only', True
        )
        kwargs['_preload_content'] = kwargs.get(
            '_preload_content', True
        )
        kwargs['_request_timeout'] = kwargs.get(
            '_request_timeout', None
        )
        kwargs['_check_input_type'] = kwargs.get(
            '_check_input_type', True
        )
        kwargs['_check_return_type'] = kwargs.get(
            '_check_return_type', True
        )
        kwargs['_spec_property_naming'] = kwargs.get(
            '_spec_property_naming', False
        )
        kwargs['_content_type'] = kwargs.get(
            '_content_type')
        kwargs['_host_index'] = kwargs.get('_host_index')
        kwargs['uid'] = \
            uid
        return self.paper_get_endpoint.call_with_http_info(**kwargs)

