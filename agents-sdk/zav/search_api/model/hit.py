"""
    Search

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: latest
    Generated by: https://openapi-generator.tech
"""


import re  # noqa: F401
import sys  # noqa: F401

from zav.search_api.model_utils import (  # noqa: F401
    ApiTypeError,
    ModelComposed,
    ModelNormal,
    ModelSimple,
    cached_property,
    change_keys_js_to_python,
    convert_js_args_to_python_args,
    date,
    datetime,
    file_type,
    none_type,
    validate_get_composed_info,
    OpenApiModel
)
from zav.search_api.exceptions import ApiAttributeError


def lazy_import():
    from zav.search_api.model.document_type_string import DocumentTypeString
    from zav.search_api.model.git_hub_repo import GitHubRepo
    from zav.search_api.model.hit_metadata import HitMetadata
    from zav.search_api.model.private_doc_status import PrivateDocStatus
    from zav.search_api.model.private_doc_status_code import PrivateDocStatusCode
    from zav.search_api.model.representations import Representations
    from zav.search_api.model.resources import Resources
    from zav.search_api.model.tweet import Tweet
    from zav.search_api.model.uuid_string import UUIDString
    globals()['DocumentTypeString'] = DocumentTypeString
    globals()['GitHubRepo'] = GitHubRepo
    globals()['HitMetadata'] = HitMetadata
    globals()['PrivateDocStatus'] = PrivateDocStatus
    globals()['PrivateDocStatusCode'] = PrivateDocStatusCode
    globals()['Representations'] = Representations
    globals()['Resources'] = Resources
    globals()['Tweet'] = Tweet
    globals()['UUIDString'] = UUIDString


class Hit(ModelNormal):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Attributes:
      allowed_values (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          with a capitalized key describing the allowed value and an allowed
          value. These dicts store the allowed enum values.
      attribute_map (dict): The key is attribute name
          and the value is json key in definition.
      discriminator_value_class_map (dict): A dict to go from the discriminator
          variable value to the discriminator class name.
      validations (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          that stores validations for max_length, min_length, max_items,
          min_items, exclusive_maximum, inclusive_maximum, exclusive_minimum,
          inclusive_minimum, and regex.
      additional_properties_type (tuple): A tuple of classes accepted
          as additional properties values.
    """

    allowed_values = {
    }

    validations = {
    }

    @cached_property
    def additional_properties_type():
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
        lazy_import()
        return (bool, date, datetime, dict, float, int, list, str, none_type,)  # noqa: E501

    _nullable = False

    @cached_property
    def openapi_types():
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded

        Returns
            openapi_types (dict): The key is attribute name
                and the value is attribute type.
        """
        lazy_import()
        return {
            'id': (str,),  # noqa: E501
            'highlight': (str,),  # noqa: E501
            'document_type': (DocumentTypeString,),  # noqa: E501
            'score': (float,),  # noqa: E501
            'uid': (str,),  # noqa: E501
            'guid': (str, none_type,),  # noqa: E501
            'uri': (str, none_type,),  # noqa: E501
            'metadata': (HitMetadata,),  # noqa: E501
            'organize_doc_id': (str,),  # noqa: E501
            'highlight_tokens': ([str],),  # noqa: E501
            'representations': (Representations,),  # noqa: E501
            'no_references': (int,),  # noqa: E501
            'no_citations': (int,),  # noqa: E501
            'h_index_sum': (int,),  # noqa: E501
            'h_index_avg': (float,),  # noqa: E501
            'twitter_popularity_score': (int,),  # noqa: E501
            'github_score': (int,),  # noqa: E501
            'duplicates': ([Hit],),  # noqa: E501
            'tweets': ([Tweet],),  # noqa: E501
            'github_repos': ([GitHubRepo],),  # noqa: E501
            'resources': (Resources,),  # noqa: E501
            'status': (PrivateDocStatus,),  # noqa: E501
            'status_codes': ([PrivateDocStatusCode],),  # noqa: E501
            'sharing': ([str],),  # noqa: E501
            'owner_uuid': (UUIDString,),  # noqa: E501
            'status_message': (str, none_type,),  # noqa: E501
            'get_bibtex_id': (str,),  # noqa: E501
            'get_similar_docs_id': (str,),  # noqa: E501
            'get_cites_id': (str,),  # noqa: E501
            'get_refs_id': (str,),  # noqa: E501
            'share_uri': (str,),  # noqa: E501
        }

    @cached_property
    def discriminator():
        return None


    attribute_map = {
        'id': '_id',  # noqa: E501
        'highlight': 'highlight',  # noqa: E501
        'document_type': 'document_type',  # noqa: E501
        'score': 'score',  # noqa: E501
        'uid': 'uid',  # noqa: E501
        'guid': 'guid',  # noqa: E501
        'uri': 'uri',  # noqa: E501
        'metadata': 'metadata',  # noqa: E501
        'organize_doc_id': 'organize_doc_id',  # noqa: E501
        'highlight_tokens': 'highlight_tokens',  # noqa: E501
        'representations': 'representations',  # noqa: E501
        'no_references': 'no_references',  # noqa: E501
        'no_citations': 'no_citations',  # noqa: E501
        'h_index_sum': 'h_index_sum',  # noqa: E501
        'h_index_avg': 'h_index_avg',  # noqa: E501
        'twitter_popularity_score': 'twitter_popularity_score',  # noqa: E501
        'github_score': 'github_score',  # noqa: E501
        'duplicates': 'duplicates',  # noqa: E501
        'tweets': 'tweets',  # noqa: E501
        'github_repos': 'github_repos',  # noqa: E501
        'resources': 'resources',  # noqa: E501
        'status': 'status',  # noqa: E501
        'status_codes': 'status_codes',  # noqa: E501
        'sharing': 'sharing',  # noqa: E501
        'owner_uuid': 'owner_uuid',  # noqa: E501
        'status_message': 'status_message',  # noqa: E501
        'get_bibtex_id': 'get_bibtex_id',  # noqa: E501
        'get_similar_docs_id': 'get_similar_docs_id',  # noqa: E501
        'get_cites_id': 'get_cites_id',  # noqa: E501
        'get_refs_id': 'get_refs_id',  # noqa: E501
        'share_uri': 'share_uri',  # noqa: E501
    }

    read_only_vars = {
    }

    _composed_schemas = {}

    @classmethod
    @convert_js_args_to_python_args
    def _from_openapi_data(cls, id, highlight, document_type, score, uid, guid, uri, metadata, organize_doc_id, *args, **kwargs):  # noqa: E501
        """Hit - a model defined in OpenAPI

        Args:
            id (str):
            highlight (str):
            document_type (DocumentTypeString):
            score (float):
            uid (str):
            guid (str, none_type):
            uri (str, none_type):
            metadata (HitMetadata):
            organize_doc_id (str):

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            highlight_tokens ([str]): [optional]  # noqa: E501
            representations (Representations): [optional]  # noqa: E501
            no_references (int): [optional]  # noqa: E501
            no_citations (int): [optional]  # noqa: E501
            h_index_sum (int): [optional]  # noqa: E501
            h_index_avg (float): [optional]  # noqa: E501
            twitter_popularity_score (int): [optional]  # noqa: E501
            github_score (int): [optional]  # noqa: E501
            duplicates ([Hit]): [optional]  # noqa: E501
            tweets ([Tweet]): [optional]  # noqa: E501
            github_repos ([GitHubRepo]): [optional]  # noqa: E501
            resources (Resources): [optional]  # noqa: E501
            status (PrivateDocStatus): [optional]  # noqa: E501
            status_codes ([PrivateDocStatusCode]): [optional]  # noqa: E501
            sharing ([str]): [optional]  # noqa: E501
            owner_uuid (UUIDString): [optional]  # noqa: E501
            status_message (str, none_type): [optional]  # noqa: E501
            get_bibtex_id (str): [optional]  # noqa: E501
            get_similar_docs_id (str): [optional]  # noqa: E501
            get_cites_id (str): [optional]  # noqa: E501
            get_refs_id (str): [optional]  # noqa: E501
            share_uri (str): [optional]  # noqa: E501
        """

        _check_type = kwargs.pop('_check_type', True)
        _spec_property_naming = kwargs.pop('_spec_property_naming', False)
        _path_to_item = kwargs.pop('_path_to_item', ())
        _configuration = kwargs.pop('_configuration', None)
        _visited_composed_classes = kwargs.pop('_visited_composed_classes', ())

        self = super(OpenApiModel, cls).__new__(cls)

        if args:
            raise ApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments." % (
                    args,
                    self.__class__.__name__,
                ),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.id = id
        self.highlight = highlight
        self.document_type = document_type
        self.score = score
        self.uid = uid
        self.guid = guid
        self.uri = uri
        self.metadata = metadata
        self.organize_doc_id = organize_doc_id
        for var_name, var_value in kwargs.items():
            if var_name not in self.attribute_map and \
                        self._configuration is not None and \
                        self._configuration.discard_unknown_keys and \
                        self.additional_properties_type is None:
                # discard variable.
                continue
            setattr(self, var_name, var_value)
        return self

    required_properties = set([
        '_data_store',
        '_check_type',
        '_spec_property_naming',
        '_path_to_item',
        '_configuration',
        '_visited_composed_classes',
    ])

    @convert_js_args_to_python_args
    def __init__(self, id, highlight, document_type, score, uid, guid, uri, metadata, organize_doc_id, *args, **kwargs):  # noqa: E501
        """Hit - a model defined in OpenAPI

        Args:
            id (str):
            highlight (str):
            document_type (DocumentTypeString):
            score (float):
            uid (str):
            guid (str, none_type):
            uri (str, none_type):
            metadata (HitMetadata):
            organize_doc_id (str):

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            highlight_tokens ([str]): [optional]  # noqa: E501
            representations (Representations): [optional]  # noqa: E501
            no_references (int): [optional]  # noqa: E501
            no_citations (int): [optional]  # noqa: E501
            h_index_sum (int): [optional]  # noqa: E501
            h_index_avg (float): [optional]  # noqa: E501
            twitter_popularity_score (int): [optional]  # noqa: E501
            github_score (int): [optional]  # noqa: E501
            duplicates ([Hit]): [optional]  # noqa: E501
            tweets ([Tweet]): [optional]  # noqa: E501
            github_repos ([GitHubRepo]): [optional]  # noqa: E501
            resources (Resources): [optional]  # noqa: E501
            status (PrivateDocStatus): [optional]  # noqa: E501
            status_codes ([PrivateDocStatusCode]): [optional]  # noqa: E501
            sharing ([str]): [optional]  # noqa: E501
            owner_uuid (UUIDString): [optional]  # noqa: E501
            status_message (str, none_type): [optional]  # noqa: E501
            get_bibtex_id (str): [optional]  # noqa: E501
            get_similar_docs_id (str): [optional]  # noqa: E501
            get_cites_id (str): [optional]  # noqa: E501
            get_refs_id (str): [optional]  # noqa: E501
            share_uri (str): [optional]  # noqa: E501
        """

        _check_type = kwargs.pop('_check_type', True)
        _spec_property_naming = kwargs.pop('_spec_property_naming', False)
        _path_to_item = kwargs.pop('_path_to_item', ())
        _configuration = kwargs.pop('_configuration', None)
        _visited_composed_classes = kwargs.pop('_visited_composed_classes', ())

        if args:
            raise ApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments." % (
                    args,
                    self.__class__.__name__,
                ),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.id = id
        self.highlight = highlight
        self.document_type = document_type
        self.score = score
        self.uid = uid
        self.guid = guid
        self.uri = uri
        self.metadata = metadata
        self.organize_doc_id = organize_doc_id
        for var_name, var_value in kwargs.items():
            if var_name not in self.attribute_map and \
                        self._configuration is not None and \
                        self._configuration.discard_unknown_keys and \
                        self.additional_properties_type is None:
                # discard variable.
                continue
            setattr(self, var_name, var_value)
            if var_name in self.read_only_vars:
                raise ApiAttributeError(f"`{var_name}` is a read-only attribute. Use `from_openapi_data` to instantiate "
                                     f"class with read only attributes.")