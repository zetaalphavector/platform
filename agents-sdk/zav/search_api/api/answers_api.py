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
from zav.search_api.model.best_answer_form import BestAnswerForm
from zav.search_api.model.best_answer_item import BestAnswerItem
from zav.search_api.model.explain_form import ExplainForm
from zav.search_api.model.explain_item import ExplainItem
from zav.search_api.model.generic_error import GenericError
from zav.search_api.model.k_answers_form import KAnswersForm
from zav.search_api.model.k_answers_item import KAnswersItem
from zav.search_api.model.uuid_string import UUIDString


class AnswersApi(object):
    """NOTE: This class is auto generated by OpenAPI Generator
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client
        self.best_answer_endpoint = _Endpoint(
            settings={
                'response_type': (BestAnswerItem,),
                'auth': [],
                'endpoint_path': '/answers/best',
                'operation_id': 'best_answer',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'best_answer_form',
                    'requester_uuid',
                    'user_roles',
                    'tenant',
                    'index_cluster',
                ],
                'required': [
                    'best_answer_form',
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
                    'best_answer_form':
                        (BestAnswerForm,),
                    'requester_uuid':
                        (UUIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                    'index_cluster':
                        (str,),
                },
                'attribute_map': {
                    'requester_uuid': 'requester-uuid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                    'index_cluster': 'index_cluster',
                },
                'location_map': {
                    'best_answer_form': 'body',
                    'requester_uuid': 'header',
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
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.explain_endpoint = _Endpoint(
            settings={
                'response_type': (ExplainItem,),
                'auth': [],
                'endpoint_path': '/answers/explain',
                'operation_id': 'explain',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'explain_form',
                    'requester_uuid',
                    'user_roles',
                    'tenant',
                    'index_cluster',
                ],
                'required': [
                    'explain_form',
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
                    'explain_form':
                        (ExplainForm,),
                    'requester_uuid':
                        (UUIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                    'index_cluster':
                        (str,),
                },
                'attribute_map': {
                    'requester_uuid': 'requester-uuid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                    'index_cluster': 'index_cluster',
                },
                'location_map': {
                    'explain_form': 'body',
                    'requester_uuid': 'header',
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
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )
        self.k_answers_endpoint = _Endpoint(
            settings={
                'response_type': (KAnswersItem,),
                'auth': [],
                'endpoint_path': '/answers/k',
                'operation_id': 'k_answers',
                'http_method': 'POST',
                'servers': None,
            },
            params_map={
                'all': [
                    'k_answers_form',
                    'requester_uuid',
                    'user_roles',
                    'tenant',
                    'index_cluster',
                ],
                'required': [
                    'k_answers_form',
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
                    'k_answers_form':
                        (KAnswersForm,),
                    'requester_uuid':
                        (UUIDString,),
                    'user_roles':
                        (str,),
                    'tenant':
                        (str,),
                    'index_cluster':
                        (str,),
                },
                'attribute_map': {
                    'requester_uuid': 'requester-uuid',
                    'user_roles': 'user-roles',
                    'tenant': 'tenant',
                    'index_cluster': 'index_cluster',
                },
                'location_map': {
                    'k_answers_form': 'body',
                    'requester_uuid': 'header',
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
                'content_type': [
                    'application/json'
                ]
            },
            api_client=api_client
        )

    def best_answer(
        self,
        best_answer_form,
        **kwargs
    ):
        """Get an answer from a question and a selection of document ids  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.best_answer(best_answer_form, async_req=True)
        >>> result = thread.get()

        Args:
            best_answer_form (BestAnswerForm):

        Keyword Args:
            requester_uuid (UUIDString): [optional]
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
            BestAnswerItem
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
        kwargs['best_answer_form'] = \
            best_answer_form
        return self.best_answer_endpoint.call_with_http_info(**kwargs)

    def explain(
        self,
        explain_form,
        **kwargs
    ):
        """Get an explanation from a passage of text and a selection of document ids  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.explain(explain_form, async_req=True)
        >>> result = thread.get()

        Args:
            explain_form (ExplainForm):

        Keyword Args:
            requester_uuid (UUIDString): [optional]
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
            ExplainItem
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
        kwargs['explain_form'] = \
            explain_form
        return self.explain_endpoint.call_with_http_info(**kwargs)

    def k_answers(
        self,
        k_answers_form,
        **kwargs
    ):
        """Get an answer from a question and a document id  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True

        >>> thread = api.k_answers(k_answers_form, async_req=True)
        >>> result = thread.get()

        Args:
            k_answers_form (KAnswersForm):

        Keyword Args:
            requester_uuid (UUIDString): [optional]
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
            KAnswersItem
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
        kwargs['k_answers_form'] = \
            k_answers_form
        return self.k_answers_endpoint.call_with_http_info(**kwargs)

