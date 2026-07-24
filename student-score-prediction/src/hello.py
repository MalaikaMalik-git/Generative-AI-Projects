"""
First program for the Student Score Prediction System.
Verifies the environment is working end-to-end.
"""


def greet(name: str) -> str:
    """Return a friendly greeting for the given name."""
    return (
        f"Hello, {name}! Environment is ready for the Student Score Prediction project."
    )


if __name__ == "__main__":
    print(greet("Malaika"))
