class TradingBotException(Exception):
    pass

class NetworkError(TradingBotException):
    pass

class DataError(TradingBotException):
    pass

class ValidationError(TradingBotException):
    pass

class APIRateLimitError(NetworkError):
    pass

class InvalidSymbolError(ValidationError):
    pass

class InsufficientDataError(DataError):
    pass

class ModelError(TradingBotException):
    pass

class ConfigurationError(TradingBotException):
    pass

