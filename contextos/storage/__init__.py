"""Durable storage backends for ContextOS."""

from contextos.storage.swap import SwapStorage, SwapStorageError

__all__ = ["SwapStorage", "SwapStorageError"]
