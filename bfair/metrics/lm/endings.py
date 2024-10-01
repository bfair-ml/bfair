import re

TEST = "... señor/a ...\n... profesor/a ...\n... amigo/a ...\n... doctor/a ..."

# jefe/a - amigo/a
PATTERN_EO_A = re.compile(r"(\b\w+?)(e|o)/a(\s*[\.,;:\?!]?)")
REPLACEMENT_EO_A = r"\1\2 o \1a\3"

# doctor/a
PATTERN_R_RA = re.compile(r"(\b\w+?)r/a(\s*[\.,;:\?!]?)")
REPLACEMENT_EO_A = r"\1r o \1ra\2"

# jefe/a - amigo/a - doctor/a
PATTERN_JOINT = re.compile(r"(\b\w+?)(e|o|r)/a(\s*[\.,;:\?!]?)")


def replacement_joint(m):
    return f"{m[1]}{m[2]} {m[1]}{'ra' if m[2] == 'r' else 'a'}{m[3]}"


def spanish_split_gender_endings(text):
    return re.sub(PATTERN_JOINT, replacement_joint, text)
