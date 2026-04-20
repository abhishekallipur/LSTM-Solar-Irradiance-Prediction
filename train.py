import sys

from lstm_model import main


if __name__ == "__main__":
    # Automation entrypoint: disable interactive windows unless explicitly overridden.
    if "--no-plot" not in sys.argv:
        sys.argv.append("--no-plot")
    main()
