python scripts/balance_villanos.py --objective Mentions
python scripts/balance_villanos.py --objective Targets
python scripts/debias_villanos.py --objective Mentions
python scripts/debias_villanos.py --objective Targets


# {1: 19, 0: 38, 2: 38}
# no 499
# [] 156 -> 38
# ['female'] 273 -> 19
# ['male'] 32 -> 19
# ['male', 'female'] 38 -> 38
# yes 499
# [] 125 -> 38
# ['female'] 252 -> 19
# ['male'] 19 -> 19
# ['male', 'female'] 103 -> 38


# {0: 20, 1: 19, 2: 2}
# no 499
# [] 330 -> 20
# ['female'] 148 -> 19
# ['male'] 19 -> 19
# ['male', 'female'] 2 -> 2
# yes 499
# [] 252 -> 20
# ['female'] 180 -> 19
# ['male'] 40 -> 19
# ['male', 'female'] 27 -> 2


# {('female',): 252, (): 125, ('male', 'female'): 38, ('male',): 19}
# no 499
# [] 156 -> 125
# ['female'] 273 -> 252
# ['male'] 32 -> 19
# ['male', 'female'] 38 -> 38
# yes 499
# [] 125 -> 125
# ['female'] 252 -> 252
# ['male'] 19 -> 19
# ['male', 'female'] 103 -> 38


# {(): 252, ('female',): 148, ('male',): 19, ('male', 'female'): 2}
# no 499
# [] 330 -> 252
# ['female'] 148 -> 148
# ['male'] 19 -> 19
# ['male', 'female'] 2 -> 2
# yes 499
# [] 252 -> 252
# ['female'] 180 -> 148
# ['male'] 40 -> 19
# ['male', 'female'] 27 -> 2