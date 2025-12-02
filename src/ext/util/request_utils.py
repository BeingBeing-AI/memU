from contextvars import ContextVar
from typing import Optional

user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


def set_current_user_id(user_id: str | None):
    """
    Set the current user ID in the context variable.

    Args:
        user_id (str): The user ID to set
    """
    user_id_ctx.set(user_id)

def get_current_user_id() -> Optional[str]:
    """
    Get the current user ID from the context variable.

    Returns:
        Optional[str]: The current user ID, or None if not set
    """
    return user_id_ctx.get()
