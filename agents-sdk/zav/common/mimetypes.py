import mimetypes


def register_mime_types() -> None:
    """Patch the process-wide ``mimetypes`` registry with types the stdlib gets
    wrong or omits.

    Must be called once at import time by any module that relies on extension
    <-> MIME resolution; it replaces the previous import-side-effect so the
    registration is explicit. Idempotent — ``mimetypes.add_type`` overwrites the
    existing mapping, so repeated calls are safe.
    """
    # Python returns "model/JT" (uppercase) but IANA standard and our system
    # use "model/jt".
    mimetypes.add_type("model/jt", ".jt")
    # Python's mimetypes module does not know the Outlook MSG type at all.
    mimetypes.add_type("application/vnd.ms-outlook", ".msg")
    mimetypes.add_type("message/rfc822", ".eml")
    # Python knows .rtf only on some platforms; register it for cross-platform safety.
    mimetypes.add_type("application/rtf", ".rtf")
    # Python's mimetypes module does not know the modern OOXML Visio family.
    mimetypes.add_type("application/vnd.ms-visio.drawing", ".vsdx")
    mimetypes.add_type("application/vnd.ms-visio.drawing.macroEnabled", ".vsdm")
    mimetypes.add_type("application/vnd.ms-visio.stencil", ".vssx")
    mimetypes.add_type("application/vnd.ms-visio.template", ".vstx")
