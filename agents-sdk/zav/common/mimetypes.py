import mimetypes

# Register correct MIME types that Python's mimetypes module has wrong or missing.
# Python returns "model/JT" (uppercase) but IANA standard and our system use "model/jt".
mimetypes.add_type("model/jt", ".jt")
