"""
Utility functions for SCOPE.
"""
import sys


def get_file_lock_functions():
    """
    Get platform-appropriate file locking functions.
    
    Returns:
        Tuple of (lock_file, unlock_file) functions
    
    Usage:
        lock_file, unlock_file = get_file_lock_functions()
        with open('file.txt', 'r+') as f:
            lock_file(f)
            try:
                # Do file operations
            finally:
                unlock_file(f)
    """
    if sys.platform == 'win32':
        # Windows-specific locking
        import msvcrt

        def lock_file(f):
            """Acquire exclusive lock on file (Windows)."""
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                # If non-blocking fails, try blocking
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

        def unlock_file(f):
            """Release lock on file (Windows)."""
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass  # Ignore unlock errors
    else:
        # Unix/Linux/Mac locking
        import fcntl

        def lock_file(f):
            """Acquire exclusive lock on file (Unix)."""
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        def unlock_file(f):
            """Release lock on file (Unix)."""
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return lock_file, unlock_file


# Pre-initialize the lock functions for convenience
lock_file, unlock_file = get_file_lock_functions()

