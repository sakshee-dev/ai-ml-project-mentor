import atexit
import logging
import os
import sys
import time
from pathlib import Path
import psutil
import msvcrt


class Lock:
    """
    Backward-compatible lock implementation.
    Behavior:
    - Uses OS-level lock on a SEPARATE file (lock.hlock) to avoid locking lock.txt
    - lock.txt stays readable while app is running (VS Code / monitoring app)
    - lock.txt first line remains plain PID for existing trackers
    - lock folder stays ./lock/<namespace> relative to working directory (original behavior)
    """

    def __init__(self, namespace: str):
        self.logger = logging.getLogger("log_console_file")
        self.namespace = (namespace or "").strip()

        self.lock_folder_name = ""
        self._lock_folder_path = None
        self._lock_txt_path = None
        self._lock_handle_path = None

        self._lock_handle_fh = None  # file handle kept open for OS lock

        self.check_lock_dir()
        atexit.register(self.release)

    def check_lock_dir(self):
        lock_dir = Path("./lock") / self.namespace if self.namespace else Path("./lock")

        self._lock_folder_path = lock_dir
        self._lock_txt_path = lock_dir / "lock.txt"          # readable PID file
        self._lock_handle_path = lock_dir / "lock.hlock"     # actual locked handle file
        self.lock_folder_name = str(lock_dir)

        if not lock_dir.exists():
            self.logger.info(f"The directory {lock_dir} does not exist. It is created")
            lock_dir.mkdir(parents=True, exist_ok=True)

    def check_lock_file(self) -> bool:
        """
        Returns:
        - True  => busy (another instance running)
        - False => not busy (this instance acquired lock)
        """
        if self._lock_folder_path is None or self._lock_txt_path is None or self._lock_handle_path is None:
            self.check_lock_dir()

        acquired = self._acquire_os_lock_handle()
        if acquired:
            self._write_lock_txt_best_effort()

        return not acquired

    def _acquire_os_lock_handle(self) -> bool:
        """
        Acquire atomic OS lock on lock.hlock (NOT on lock.txt).
        This avoids blocking readers of lock.txt.
        """
        try:
            # Open/create the handle file and keep it open
            self._lock_handle_fh = open(self._lock_handle_path, "a+b")
        except Exception as e:
            self.logger.error(f"Lock handle file cannot be opened: {self._lock_handle_path}. Error: {e}")
            return False

        try:
            # Lock 1 byte in handle file, non-blocking
            self._lock_handle_fh.seek(0)
            msvcrt.locking(self._lock_handle_fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            # Another instance holds the lock
            other_pid = self._read_pid_best_effort()
            if other_pid:
                self.logger.error(f"Another instance is already running (lock held). pid={other_pid}")
            else:
                self.logger.error("Another instance is already running (lock held).")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while acquiring lock: {e}")
            return False

    def _write_lock_txt_best_effort(self) -> None:
        try:
            pid = os.getpid()
            with open(self._lock_txt_path, "w", encoding="utf-8") as f:
                f.write(str(pid))
        except Exception as e:
            self.logger.warning(f"Failed to write lock.txt pid: {e}")

    def _read_pid_best_effort(self) -> int | None:
        """
        Reads first line of lock.txt and parses PID.
        """
        try:
            if not self._lock_txt_path or not self._lock_txt_path.exists():
                return None
            with open(self._lock_txt_path, "r", encoding="utf-8") as f:
                first = (f.readline() or "").strip()
            if not first:
                return None
            if first.lower().startswith("pid="):
                first = first.split("=", 1)[1].strip()
            return int(first)
        except Exception:
            return None

    def release(self) -> None:
        """Release OS-level lock (safe to call multiple times)."""
        if not self._lock_handle_fh:
            return

        try:
            self._lock_handle_fh.seek(0)
            msvcrt.locking(self._lock_handle_fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass

        try:
            self._lock_handle_fh.close()
        except Exception:
            pass

        self._lock_handle_fh = None

    def check_pid_running(self, pid: int) -> bool:
        """Compatibility helper."""
        try:
            p = psutil.Process(pid)
            return p.is_running()
        except Exception:
            return False
