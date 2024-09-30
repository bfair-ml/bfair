from bfair.utils.spacy_trf_vecs import PersonCheckerForSpanish

checker = PersonCheckerForSpanish()
nlp = checker.nlp

music = [
    (nlp("Mi amiga trabaja como música.")[-2], True),
    (nlp("La música relaja.")[1], False),
    (nlp("En la música no importa la letra.")[2], False),
    (nlp("La música que vive debajo de mi casa canta muy bien.")[1], True),
    (nlp("La música me encanta.")[1], False),
    (nlp("A ella le encanta la música.")[-2], False),
]

politics = [
    (nlp("La política es una rama de ...")[2], False),
    (nlp("No me gusta la política de la que hablan.")[4], False),
    (nlp("No me gusta la política.")[-2], False),
    (nlp("Mi vecina trabaja como política.")[-2], True),
    (nlp("La política que da un discurso al frente es increible.")[1], True),
]

for word, gold in music + politics:
    prediction = checker.check_token(word)
    if prediction != gold:
        print("⚠️", word.sent)
        print("Expected: {}. Got: {}.".format(gold, prediction))
    else:
        print("✅", word.sent)
