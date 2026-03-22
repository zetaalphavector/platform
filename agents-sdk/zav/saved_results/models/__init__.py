# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from zav.saved_results.model.annotation_highlight_form import AnnotationHighlightForm
from zav.saved_results.model.annotation_highlight_item import AnnotationHighlightItem
from zav.saved_results.model.annotation_highlight_rect import AnnotationHighlightRect
from zav.saved_results.model.default_user_order import DefaultUserOrder
from zav.saved_results.model.favorite_tag_form import FavoriteTagForm
from zav.saved_results.model.favorite_tag_item import FavoriteTagItem
from zav.saved_results.model.favorite_tag_item_settings import FavoriteTagItemSettings
from zav.saved_results.model.favorite_tag_item_stats import FavoriteTagItemStats
from zav.saved_results.model.followed_tag_form import FollowedTagForm
from zav.saved_results.model.followed_tag_item import FollowedTagItem
from zav.saved_results.model.followed_tag_item_settings import FollowedTagItemSettings
from zav.saved_results.model.followed_tag_item_stats import FollowedTagItemStats
from zav.saved_results.model.followed_tag_items import FollowedTagItems
from zav.saved_results.model.guid_string import GUIDString
from zav.saved_results.model.generic_error import GenericError
from zav.saved_results.model.generic_tag_item import GenericTagItem
from zav.saved_results.model.generic_tag_item_settings import GenericTagItemSettings
from zav.saved_results.model.generic_tag_item_stats import GenericTagItemStats
from zav.saved_results.model.generic_tag_items import GenericTagItems
from zav.saved_results.model.new_tagged_resource_item import NewTaggedResourceItem
from zav.saved_results.model.note_export_form import NoteExportForm
from zav.saved_results.model.note_form import NoteForm
from zav.saved_results.model.note_item import NoteItem
from zav.saved_results.model.note_items import NoteItems
from zav.saved_results.model.note_object_type import NoteObjectType
from zav.saved_results.model.org_sharing_policy import OrgSharingPolicy
from zav.saved_results.model.page_params import PageParams
from zav.saved_results.model.paginated_followed_tags import PaginatedFollowedTags
from zav.saved_results.model.paginated_generic_tags import PaginatedGenericTags
from zav.saved_results.model.paginated_notes import PaginatedNotes
from zav.saved_results.model.paginated_shared_tags import PaginatedSharedTags
from zav.saved_results.model.paginated_tag_notes import PaginatedTagNotes
from zav.saved_results.model.paginated_tagged_documents import PaginatedTaggedDocuments
from zav.saved_results.model.paginated_tagged_resources import PaginatedTaggedResources
from zav.saved_results.model.paginated_tags import PaginatedTags
from zav.saved_results.model.public_sharing_policy import PublicSharingPolicy
from zav.saved_results.model.resource_permission import ResourcePermission
from zav.saved_results.model.shared_tag_item import SharedTagItem
from zav.saved_results.model.shared_tag_items import SharedTagItems
from zav.saved_results.model.sharing_policy import SharingPolicy
from zav.saved_results.model.tag_followers import TagFollowers
from zav.saved_results.model.tag_followers_form import TagFollowersForm
from zav.saved_results.model.tag_followers_item import TagFollowersItem
from zav.saved_results.model.tag_followers_items import TagFollowersItems
from zav.saved_results.model.tag_form import TagForm
from zav.saved_results.model.tag_item import TagItem
from zav.saved_results.model.tag_item_settings import TagItemSettings
from zav.saved_results.model.tag_item_stats import TagItemStats
from zav.saved_results.model.tag_items import TagItems
from zav.saved_results.model.tag_name import TagName
from zav.saved_results.model.tag_note_form import TagNoteForm
from zav.saved_results.model.tag_note_item import TagNoteItem
from zav.saved_results.model.tag_note_items import TagNoteItems
from zav.saved_results.model.tag_type import TagType
from zav.saved_results.model.taggable_resource_type import TaggableResourceType
from zav.saved_results.model.tagged_document_form import TaggedDocumentForm
from zav.saved_results.model.tagged_document_item import TaggedDocumentItem
from zav.saved_results.model.tagged_document_items import TaggedDocumentItems
from zav.saved_results.model.tagged_document_update_form import TaggedDocumentUpdateForm
from zav.saved_results.model.tagged_resource_collection import TaggedResourceCollection
from zav.saved_results.model.tagged_resource_form import TaggedResourceForm
from zav.saved_results.model.tagged_resource_items import TaggedResourceItems
from zav.saved_results.model.uuid_string import UUIDString
from zav.saved_results.model.user_order import UserOrder
from zav.saved_results.model.user_sharing_policy import UserSharingPolicy
