import logging
import subprocess
import sys
from pathlib import Path


def setup_logger() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger("ai_ml_project_mentor")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


def validate_paths(project_root: Path, logger: logging.Logger) -> Path:
    """Validate required files and return the Streamlit app path."""
    streamlit_app = project_root / "ui" / "streamlit_app.py"

    if not streamlit_app.exists():
        logger.error("Missing file: %s", streamlit_app)
        raise FileNotFoundError(f"Could not find Streamlit app: {streamlit_app}")

    return streamlit_app


def launch_streamlit(project_root: Path, streamlit_app: Path, logger: logging.Logger) -> None:
    """Launch the Streamlit application."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
    ]

    logger.info("Launching Streamlit app...")
    logger.info("Command: %s", " ".join(command))

    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    logger = setup_logger()
    project_root = Path(__file__).resolve().parent

    logger.info("=" * 60)
    logger.info("AI ML Project Mentor starting...")
    logger.info("Project root: %s", project_root)
    logger.info("Python executable: %s", sys.executable)
    logger.info("=" * 60)

    try:
        streamlit_app = validate_paths(project_root, logger)
        launch_streamlit(project_root, streamlit_app, logger)

    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        sys.exit(1)

    except subprocess.CalledProcessError as exc:
        logger.exception("Streamlit process failed with return code %s", exc.returncode)
        sys.exit(exc.returncode)

    except KeyboardInterrupt:
        logger.warning("Application stopped by user.")
        sys.exit(0)

    except Exception as exc:
        logger.exception("Unexpected error while starting application: %s", exc)
        sys.exit(1)

    finally:
        logger.info("Application shutdown complete.")


if __name__ == "__main__":
    main()