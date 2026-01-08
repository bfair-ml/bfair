import re

SPANISH_TEST = """
... señor/a ...
... profesor/a ...
... amigo/a ...
... doctor/a ...
... jefe/a ...
... señores/as ...
... profesores/as ...
... amigos/as ...
... doctores/as ...
... jefes/as ...
"""


class SpanishGenderPreprocessor:
    # jefe/a - amigo/a
    PATTERN_EO_A = re.compile(r"(\b\w+?)(e|o)/a(\s*[\.,;:\?!]?)")
    REPLACEMENT_EO_A = r"\1\2 o \1a\3"

    # doctor/a
    PATTERN_R_RA = re.compile(r"(\b\w+?)r/a(\s*[\.,;:\?!]?)")
    REPLACEMENT_R_RA = r"\1r o \1ra\2"

    # jefe/a - amigo/a - doctor/a
    PATTERN_JOINT = re.compile(r"(\b\w+?)(e|o|r|es|os)/(a|as)\b")

    @staticmethod
    def replacement_joint(m):
        # The '/' is important to avoid syntactical errors while tokenizing
        # "Un doctor/a ... " -> "Un doctor doctora ..." -> "DET NOUN **VERB**"
        return f"{m[1]}{m[2]} / {m[1]}{'r' if m[2] == 'r' else ''}{m[3]}"

    @classmethod
    def split_gender_endings(cls, text):
        return re.sub(cls.PATTERN_JOINT, cls.replacement_joint, text)


CATALAN_TEST = """
... amic/a ...
... mestre/a ...
... director/a ...
... professors/as ...
... directors/as ...
... mestres/as ...
"""


class CatalanGenderPreprocessor:
    # Rule: -or/a → or o ora (e.g., director/a)
    PATTERN_OR_A = re.compile(r"(\b\w+?)or/a(\b|\s*[\.,;:\?!])")
    REPLACEMENT_OR_A = r"\1or o \1ora\2"

    # Rule: -r/a → r o ra (e.g., professor/a)
    PATTERN_R_A = re.compile(r"(\b\w+?)r/a(\b|\s*[\.,;:\?!])")
    REPLACEMENT_R_A = r"\1r o \1ra\2"

    # Rule: -e/a → e o a (e.g., mestre/a)
    PATTERN_E_A = re.compile(r"(\b\w+?)e/a(\b|\s*[\.,;:\?!])")
    REPLACEMENT_E_A = r"\1e o \1a\2"

    # Rule: -o/a → o o a (e.g., amic/a)
    PATTERN_O_A = re.compile(r"(\b\w+?)o/a(\b|\s*[\.,;:\?!])")
    REPLACEMENT_O_A = r"\1o o \1a\2"

    # Rule: -s/as → s o sas (e.g., mestres/as)
    PATTERN_S_AS = re.compile(r"(\b\w+?)s/as(\b|\s*[\.,;:\?!])")
    REPLACEMENT_S_AS = r"\1s o \1ses\2"  # crude plural feminine, could be smarter

    # Optional: catch generic /(a|as) endings
    PATTERN_GENERIC = re.compile(r"(\b\w+?)/([a|as])(\b|\s*[\.,;:\?!])")
    REPLACEMENT_GENERIC = r"\1 o \1\2\3"

    # Function to apply all patterns
    @classmethod
    def split_gender_endings(cls, text: str) -> str:
        print(
            "⚠️ Warning: Catalan version is not well tuned and may produce inaccurate results."
        )
        text = cls.PATTERN_OR_A.sub(cls.REPLACEMENT_OR_A, text)
        text = cls.PATTERN_R_A.sub(cls.REPLACEMENT_R_A, text)
        text = cls.PATTERN_E_A.sub(cls.REPLACEMENT_E_A, text)
        text = cls.PATTERN_O_A.sub(cls.REPLACEMENT_O_A, text)
        text = cls.PATTERN_S_AS.sub(cls.REPLACEMENT_S_AS, text)
        text = cls.PATTERN_GENERIC.sub(cls.REPLACEMENT_GENERIC, text)  # fallback
        return text


if __name__ == "__main__":
    print(SPANISH_TEST)
    print("------------")
    print(SpanishGenderPreprocessor.split_gender_endings(SPANISH_TEST))
    print("============")
    print(CATALAN_TEST)
    print("------------")
    print(CatalanGenderPreprocessor.split_gender_endings(CATALAN_TEST))
