import re

TEST = """
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

# jefe/a - amigo/a
PATTERN_EO_A = re.compile(r"(\b\w+?)(e|o)/a(\s*[\.,;:\?!]?)")
REPLACEMENT_EO_A = r"\1\2 o \1a\3"

# doctor/a
PATTERN_R_RA = re.compile(r"(\b\w+?)r/a(\s*[\.,;:\?!]?)")
REPLACEMENT_EO_A = r"\1r o \1ra\2"

# jefe/a - amigo/a - doctor/a
PATTERN_JOINT = re.compile(r"(\b\w+?)(e|o|r|es|os)/(a|as)\b")


def replacement_joint(m):
    # The '/'  is important to avoid sintactical errors while tokenizing
    # "Un doctor/a ... " -> "Un doctor doctora ..." -> "DET NOUN **VERB**"
    return f"{m[1]}{m[2]} / {m[1]}{'r' if m[2] == 'r' else ''}{m[3]}"


def spanish_split_gender_endings(text):
    return re.sub(PATTERN_JOINT, replacement_joint, text)


if __name__ == "__main__":
    print(TEST)
    print("------------")
    print(spanish_split_gender_endings(TEST))
