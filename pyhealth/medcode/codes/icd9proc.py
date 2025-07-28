from pyhealth.medcode.inner_map import InnerMap


# TODO: add convert


class ICD9PROC(InnerMap):
    """9-th International Classification of Diseases, Procedure."""

    def __init__(self, **kwargs):
        super(ICD9PROC, self).__init__(vocabulary="ICD9PROC", **kwargs)

    @staticmethod
    def standardize(code: str):
        """Standardizes ICD9PROC code."""
        # Most codes are already standardized or short.
        # Fast path: if the code contains a dot or is very short, return as-is
        if "." in code or len(code) <= 2:
            return code
        # Otherwise, insert the dot after the second character
        return f"{code[:2]}.{code[2:]}"


if __name__ == "__main__":
    code_sys = ICD9PROC(refresh_cache=True)
    code_sys.stat()
    print("81.01" in code_sys)
    print(code_sys.lookup("01.31"))
    print(code_sys.get_ancestors("01.31"))
    print(code_sys.get_descendants("01"))
