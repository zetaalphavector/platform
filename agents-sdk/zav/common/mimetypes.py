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
    # Python's mimetypes module does not know the OneNote section type.
    mimetypes.add_type("application/onenote", ".one")
    # Python's mimetypes module does not know the modern OOXML Visio family.
    mimetypes.add_type("application/vnd.ms-visio.drawing", ".vsdx")
    mimetypes.add_type("application/vnd.ms-visio.drawing.macroEnabled", ".vsdm")
    mimetypes.add_type("application/vnd.ms-visio.stencil", ".vssx")
    mimetypes.add_type("application/vnd.ms-visio.template", ".vstx")
    # CAD formats; none are known to the stdlib. .prt and .asm are absent because
    # they are ambiguous (Creo vs NX); user-documents-connector disambiguates them
    # by binary header at runtime.
    mimetypes.add_type("model/step", ".step")
    mimetypes.add_type("model/step", ".stp")
    mimetypes.add_type("model/acis", ".sat")
    mimetypes.add_type("model/acis", ".sab")
    mimetypes.add_type("model/catia-v4", ".model")
    mimetypes.add_type("model/catia-v4", ".dlv")
    mimetypes.add_type("model/catia-v4", ".exp")
    mimetypes.add_type("model/catia-v4", ".session")
    # guess_type lowercases before lookup, so only the lowercase spellings are
    # reachable; the CATIA-cased pair is registered first so guess_extension keeps it.
    mimetypes.add_type("model/catia-v5", ".CATPart")
    mimetypes.add_type("model/catia-v5", ".CATProduct")
    mimetypes.add_type("model/catia-v5", ".catpart")
    mimetypes.add_type("model/catia-v5", ".catproduct")
    mimetypes.add_type("model/3dxml", ".3dxml")
    mimetypes.add_type("model/dwg", ".dwg")
    mimetypes.add_type("model/fusion360", ".f3d")
    mimetypes.add_type("model/iges", ".iges")
    mimetypes.add_type("model/iges", ".igs")
    mimetypes.add_type("model/inventor", ".ipt")
    mimetypes.add_type("model/inventor", ".iam")
    mimetypes.add_type("model/vnd.parasolid.transmit.text", ".x_t")
    mimetypes.add_type("model/vnd.parasolid.transmit.binary", ".x_b")
    mimetypes.add_type("model/3dm", ".3dm")
    mimetypes.add_type("model/solid-edge", ".par")
    mimetypes.add_type("model/solid-edge", ".psm")
    mimetypes.add_type("model/solidworks", ".sldprt")
    mimetypes.add_type("model/solidworks", ".sldasm")
    mimetypes.add_type("model/zw3d", ".z3")
    mimetypes.add_type("model/zw3d", ".z3prt")
    # Python maps .wav and .flac to the non-standard x- types; IANA registers
    # audio/wav and audio/flac.
    mimetypes.add_type("audio/wav", ".wav")
    mimetypes.add_type("audio/flac", ".flac")
