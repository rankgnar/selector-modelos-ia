"""Errors with a plain-Spanish message meant to be shown directly to the user."""


class SelectorError(Exception):
    """Base error. `user_message` is safe to show in the UI."""

    def __init__(self, user_message: str, *, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class OfflineError(SelectorError):
    pass


class RateLimitError(SelectorError):
    def __init__(self, user_message: str, *, retry_after: float | None = None, detail: str = ""):
        super().__init__(user_message, detail=detail)
        self.retry_after = retry_after


class InvalidKeyError(SelectorError):
    pass


class InsufficientCreditsError(SelectorError):
    pass


class NotFoundError(SelectorError):
    pass


class ApiError(SelectorError):
    pass
