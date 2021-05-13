try:
    from contextlib import nullcontext  # python > 3.7 needed for this
except ImportError:
    # Backward compatibility with python <= 3.6
    class nullcontext:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


class nullwriter:
    @staticmethod
    def flush():
        pass

    @staticmethod
    def as_default():
        return nullcontext()

