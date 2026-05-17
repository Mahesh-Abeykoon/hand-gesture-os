"""GestureOS application entry point.

By default GestureOS starts the advanced PyQt6 dashboard. If PyQt6 is not
installed, it falls back to the lightweight Tkinter dashboard.
"""

if __name__ == "__main__":
    try:
        from gestureos.ui.qt_app import main
    except Exception:
        from gestureos.ui.app import main
    main()
