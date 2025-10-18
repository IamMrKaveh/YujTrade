class TradingBotException(Exception):
    """Base exception for all custom exceptions in the application."""

    pass


class NetworkError(TradingBotException):
    """Raised for network-related issues like connection errors."""

    pass


class DataError(TradingBotException):
    """Base exception for data-related problems."""

    pass


class ValidationError(DataError):
    """Raised when data validation fails."""

    pass


class APIRateLimitError(NetworkError):
    """Raised when an API rate limit is exceeded."""

    pass


class InvalidSymbolError(ValidationError):
    """
    Raised when a symbol is not supported by an exchange or API.
    This error should typically not be retried.
    """

    pass


class InsufficientDataError(DataError):
    """Raised when there is not enough data to perform an operation."""

    pass


class ModelError(TradingBotException):
    """Raised for errors related to machine learning models."""

    pass


class ConfigurationError(TradingBotException):
    """Raised for configuration-related problems."""

    pass


class IndicatorError(DataError):
    """Raised for errors during indicator calculation."""

    pass


class ObjectClosedError(TradingBotException):
    """Raised when an operation is attempted on a closed object."""

    pass
