import urllib.parse

GET_JSON = True


def property_of_entity(entity, prop_name):
    _uri = "http://dbpedia.org"
    _uri = urllib.parse.quote(_uri.encode("utf-8"), safe="+{}")

    _query = (
        f"SELECT+?{prop_name}+WHERE+{{dbr:{entity}+dbp:{prop_name}+?{prop_name}+.}}"
    )
    _query = urllib.parse.quote(_query.encode("utf8"), safe="+{{}}")

    _format = "json" if GET_JSON else "text/html"
    _format = urllib.parse.quote(_format.encode("utf8"), safe="+{{}}")

    print(
        f"https://dbpedia.org/sparql?default-graph-uri={_uri}&should-sponge=grab-all&query={_query}&format={_format}"
    )


def get_people_with_property(_property, offset=0, limit=10000):
    _uri = "http://dbpedia.org"
    _uri = urllib.parse.quote(_uri.encode("utf-8"), safe="+{}")

    _query = f"SELECT+?who,+?{_property}+WHERE+{{?who+a+dbo:Person+.+?who+dbp:{_property}+?{_property}+.}}+ORDER+BY+ASC(?who)+LIMIT+{limit}+OFFSET+{offset * limit}"
    _query = urllib.parse.quote(_query.encode("utf8"), safe="+{{}}")

    _format = "json" if GET_JSON else "text/html"
    _format = urllib.parse.quote(_format.encode("utf8"), safe="+{{}}")

    print(
        f"https://dbpedia.org/sparql?default-graph-uri={_uri}&should-sponge=grab-all&query={_query}&format={_format}"
    )


def get_all_of_type(_type, offset=0, limit=10000):
    _uri = "http://dbpedia.org"
    _uri = urllib.parse.quote(_uri.encode("utf-8"), safe="+{}")

    _query = f"SELECT+?who+WHERE+{{?who+a+dbo:{_type}+.}}+ORDER+BY+ASC(?who)+LIMIT+{limit}+OFFSET+{offset * limit}"
    _query = urllib.parse.quote(_query.encode("utf8"), safe="+{{}}")

    _format = "json" if GET_JSON else "text/html"
    _format = urllib.parse.quote(_format.encode("utf8"), safe="+{{}}")

    print(
        f"https://dbpedia.org/sparql?default-graph-uri={_uri}&should-sponge=grab-all&query={_query}&format={_format}"
    )


property_of_entity("Scheherazade", "gender")
get_people_with_property("name")
get_all_of_type("Person", 1)

"Rihanna is basically master of the fashion universe right now, so we're naturally going to pay attention to what trends she is and isn't wearing whenever she steps out of the door (or black SUV). She's having quite the epic week, first presenting her Savage x Fenty lingerie runway show then hosting her annual Diamond Ball charity event last night. Rihanna was decked out in Givenchy for the big event, but upon arrival at the venue, she wore a T-shirt, diamonds (naturally), and a scarf, leather pants, and heels in fall's biggest color trend: pistachio green."
"Lisa Marie Scott is a human."