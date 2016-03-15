from pickling import *
import re

#replace URLs with "URLLINK"
def replaceURLs(token):
    return re.sub(r"http\S+", "URLLINK", token)

#replace URLs with "http"
def replaceURLs2(token):
    return re.sub(r"http\S+", "http", token)


#replace user mentions with "USERMENTION"
def replaceUserMentions(token):
    return re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", token)

#replace positive emojis with "POSITIVE_EMOJI"
def replacePositiveEmoji(token):
    result = re.sub("(\:\))|((?i):D)|((?!):P)|(;\))|(^_^)", "POSITIVE_EMOJI", token)
    return result

#replace negative emojis with "NEGATIVE_EMOJI"
def replaceNegativeEmoji(token):
    result = re.sub("(\:\()|((?i):'\()|((?!):O)", "NEGATIVE_EMOJI", token)
    return result

#capitalise contents of hashtags
def replaceHashtag(token):
    return re.sub("(?:#)([A-Za-z0-9_]+)", r"HT_\1", token)

#replace all non-alphanumeric
def replaceRest(token):
    result = re.sub("[^a-zA-Z0-9_]", "", token)
    return re.sub(' +','', result)


if __name__ == "__main__":
    print replaceURLs("http://asdasda")
    print replaceUserMentions("@Ayman")
    print replacePositiveEmoji(":)")
    print replaceNegativeEmoji(":(")
    print capHashtag("#Python")
