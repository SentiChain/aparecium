"""
Aparecium Custom Exceptions

This module defines a hierarchy of custom exceptions for Aparecium. All exceptions
inherit from the base ApareciumError, making it easier to catch any Aparecium-related
errors in a single block if desired.

Typical usage example:
    from exceptions import DatabaseError

    try:
        # Some DB operation
    except DatabaseError as exc:
        handle_db_issue(exc)
"""


class ApareciumError(Exception):
    """
    Base exception for all Aparecium-related errors.

    Inherits from the built-in Exception class. You can catch this to handle any error
    specific to Aparecium in a generic way.
    """

    pass


class DatabaseError(ApareciumError):
    """
    Raised when there are issues with database operations.

    Examples might include connection failures, read/write errors, or schema issues.
    """

    pass


class VectorizationError(ApareciumError):
    """
    Raised when there are issues with text vectorization.

    For example, if tokenization or model inference fails, this exception may be used.
    """

    pass


class ReverserError(ApareciumError):
    """
    Raised when there are issues with the reverser model or decoding process.

    This might indicate invalid model states, dimensional mismatches, or runtime
    errors during text generation.
    """

    pass


class ConfigurationError(ApareciumError):
    """
    Raised when there are issues with configuration settings.

    Common scenarios include missing environment variables, invalid file paths,
    or contradictory config parameters.
    """

    pass


class DataProcessingError(ApareciumError):
    """
    Raised when there are issues processing input data.

    This can include malformed input, unexpected data formats, or transformation errors.
    """

    pass


class APIError(ApareciumError):
    """
    Raised when there are issues with external API calls.

    This might indicate errors in network requests, unexpected API responses,
    or authentication failures.
    """

    pass
