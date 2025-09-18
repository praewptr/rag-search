import os
import subprocess
import sys


def install_requirements() -> None:
    """
    Install required packages from requirements.txt if not already installed.
    """
    try:
        print("Checking and installing required packages...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("All required packages are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)


def run_uvicorn() -> None:
    """
    Run the FastAPI application using uvicorn.
    """
    try:
        print("Starting FastAPI server with uvicorn...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ai_search_main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--reload",
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error starting FastAPI server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Install requirements
    install_requirements()

    # Run FastAPI application
    run_uvicorn()
