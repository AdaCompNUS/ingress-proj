from textblob import TextBlob

nlp_blob = TextBlob("pick up the wooden block next to the computer")

print nlp_blob.noun_phrases