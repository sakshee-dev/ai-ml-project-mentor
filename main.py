from __future__ import annotations

import asyncio
import json
import logging
import logging.config
import os
import signal
import sys
from pathlib import Path
from typing import Optional
from utils.global_variables import GlobalVariables
from utils.lock import Lock
from utils.logger_setup import setup_logging

class Main:
    """
    Main entry point for the AI ML Project Mentor application.

    Handles:
    - logger configuration
    - settings/bootstrap
    - single-instance lock
    - ctrl+c
    - graceful shutdown
    - launching Streamlit
    """

    def __init__(self, namespace: str = "") -> None:
        self.namespace = namespace or ""
        self.project_root = Path(GlobalVariables.main_path).resolve()
        self.logger_config_path = Path(GlobalVariables.log_config_path).resolve()
        self.logs_root = Path(GlobalVariables.log_path).resolve()
        self.streamlit_app_path = self.project_root / "ui" / "streamlit_app.py"

        self.logger: Optional[logging.Logger] = None
        self.is_running = True
        self.streamlit_process: Optional[asyncio.subprocess.Process] = None
        self.lock: Optional[Lock] = None

        signal.signal(signal.SIGINT, self.signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.signal_handler)

        self.read_settings()
        self.start_logger()
        self.logger = logging.getLogger(__name__)

        self.ensure_single_execution()
        self.echo_application_info()

    def start_logger(self) -> None:
        """Configure logging from logger_config.json."""
        try:
            setup_logging(
                project_root=self.project_root,
                config_path=self.logger_config_path,
                logger_name="log_console_file",
            )
            bootstrap_logger = logging.getLogger("log_console_file")
            bootstrap_logger.info("Main process logger configured successfully")

        except Exception as exc:
            print("Logger startup failed:")
            print(exc)
            raise

    def stop_logger(self) -> None:
        """Shutdown logger cleanly."""
        try:
            if self.logger:
                self.logger.info("Application terminated")
                self.logger.info("------------------------------------------------------------")
            logging.shutdown()
        except Exception as exc:
            print(f"Failed while stopping logger: {exc}")

    def echo_application_info(self) -> None:
        """Write startup details to logs."""
        try:
            app_pid = os.getpid()
            GlobalVariables.pid = app_pid

            self.logger.info("============================================================")
            self.logger.info("%s starting...", GlobalVariables.app_name)
            self.logger.info("Application version: %s", GlobalVariables.version)
            self.logger.info("Last modified date: %s", GlobalVariables.last_modified_date)
            self.logger.info("Namespace: %s", self.namespace if self.namespace else "<default>")
            self.logger.info("Project root: %s", self.project_root)
            self.logger.info("Python executable: %s", sys.executable)
            self.logger.info("Application PID: %s", app_pid)
            self.logger.info("Streamlit app path: %s", self.streamlit_app_path)
        except Exception as exc:
            if self.logger:
                self.logger.exception("Failed to echo application info: %s", exc)
            else:
                print(f"Failed to echo application info: {exc}")

    def ensure_single_execution(self) -> None:
        """
        This method ensures only one instance of this application is running
        """
        my_lock = Lock(namespace=self.namespace)
        is_busy = my_lock.check_lock_file()
        if is_busy is True:
            self.logger.error('The lock file is busy. The application is already running.')
            self.logger.info(f'Launching the {GlobalVariables.app_name} is aborted')
            self.stop_logger()  # close all logger handles
            sys.exit(GlobalVariables.LOCK_FILE_BUSY)

    def read_settings(self) -> None:
        """
        Keep this method for symmetry and future config reads.
        """
        try:
            GlobalVariables.pid = None
        except Exception as exc:
            print(f"Failed to initialize settings: {exc}")
            sys.exit(GlobalVariables.BAD_SETTING)

    def exit_gracefully(self) -> None:
        """Graceful shutdown state change."""
        try:
            self.is_running = False
            if self.logger:
                self.logger.info("Graceful shutdown initiated")
        except Exception as exc:
            if self.logger:
                self.logger.exception("Failed during graceful shutdown: %s", exc)

    def signal_handler(self, _sig, _frame) -> None:
        """Handle Ctrl+C / SIGTERM."""
        try:
            if self.logger:
                self.logger.info("Termination signal detected")
            self.exit_gracefully()
        except Exception as exc:
            print(f"Signal handler error: {exc}")

    async def launch_streamlit(self) -> int:
        """Launch the Streamlit UI subprocess."""
        try:
            if not self.streamlit_app_path.exists():
                raise FileNotFoundError(f"Streamlit app not found: {self.streamlit_app_path}")

            command = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(self.streamlit_app_path),
            ]

            self.logger.info("Launching Streamlit app...")
            self.logger.info("Command: %s", " ".join(command))

            self.streamlit_process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.project_root),
                stdout=None,
                stderr=None,
            )

            self.logger.info("Streamlit process started with pid=%s", self.streamlit_process.pid)

            return_code = await self.streamlit_process.wait()

            if return_code == 0:
                self.logger.info("Streamlit exited normally with code=%s", return_code)
            else:
                self.logger.warning("Streamlit exited with code=%s", return_code)

            return return_code

        except Exception as exc:
            self.logger.exception("Failed to launch Streamlit: %s", exc)
            raise

    async def terminate_streamlit(self) -> None:
        """Terminate Streamlit subprocess if still alive."""
        try:
            if self.streamlit_process and self.streamlit_process.returncode is None:
                self.logger.info("Terminating Streamlit process pid=%s", self.streamlit_process.pid)
                self.streamlit_process.terminate()

                try:
                    await asyncio.wait_for(self.streamlit_process.wait(), timeout=10)
                    self.logger.info("Streamlit process terminated cleanly")
                except asyncio.TimeoutError:
                    self.logger.warning("Streamlit termination timeout; killing process")
                    self.streamlit_process.kill()
                    await self.streamlit_process.wait()

        except Exception as exc:
            self.logger.exception("Failed while terminating Streamlit: %s", exc)

    async def run_event_loop(self) -> int:
        """Run the main app lifecycle."""
        exit_code = 0

        try:
            self.logger.info("run_event_loop started")
            exit_code = await self.launch_streamlit()

        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt received in run_event_loop")
            self.exit_gracefully()
            exit_code = 0

        except Exception as exc:
            self.exit_gracefully()
            self.logger.exception("run_event_loop error: %s", exc)
            exit_code = GlobalVariables.BAD_SETTING

        finally:
            await self.terminate_streamlit()

            try:
                if self.lock:
                    self.lock.release()
                    self.logger.info("Application lock released")
            except Exception as exc:
                if self.logger:
                    self.logger.exception("Failed to release lock: %s", exc)

            self.stop_logger()

        return exit_code


if __name__ == "__main__":
    namespace = ""

    try:
        if len(sys.argv) > 1:
            namespace = sys.argv[1]

        app = Main(namespace=namespace)
        exit_code = asyncio.run(app.run_event_loop())
        raise SystemExit(exit_code)

    except KeyboardInterrupt:
        print("Application stopped by user.")
        raise SystemExit(0)

    except Exception as exc:
        logging.basicConfig(level=logging.DEBUG, force=True)
        logging.getLogger(__name__).exception("Fatal error in main: %s", exc)
        raise SystemExit(1)