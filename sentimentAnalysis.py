from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="en")


def prRed(skk): return f"\033[91m{skk}\033[00m"
def prGreen(skk): return f"\033[92m{skk}\033[00m"


def sentimentAnalysis(sentence: str):
    output = analyzer.predict(sentence)
    positive = output.probas['POS']
    negative = output.probas['NEG']
    maxValue = max(positive, negative)
    if maxValue == negative:
        sentence_c = prRed(sentence)
    else:
        sentence_c = prGreen(sentence)
    if output.output == "NEG":
        out = prRed(output.output)
    else:
        out = output.output

    marklist = sorted(output.probas.items(), key=lambda x: x[1])
    marklist.reverse()
    sortdict = dict(marklist)
    for key, value in sortdict.items():
        sortdict[key] = round(value*100, 2)
    if len(sentence) > 25:
        print(f"{sentence_c:60s} {out} {output.probas[output.output] * 100:00.2f}% {sortdict}")
    else:
        print(f"{sentence_c:50s} {out} {output.probas[output.output] * 100:00.2f}% {sortdict}")


if __name__ == "__main__":
    sexualityTypes = ['heterosexual', 'homosexual', 'bisexual', 'asexual', "pansexual"]
    shortSexualityTypes = ['straight', 'gay', 'lesbian']

    for sexuality in shortSexualityTypes:
        sentimentAnalysis(f"We are living in a {sexuality} relationship.")
    print('\n')

    for sexuality in shortSexualityTypes:
        sentimentAnalysis(f"We are living in a {sexuality} marriage.")
    print('\n')

    for sexuality in sexualityTypes:
        sentimentAnalysis(f"This {sexuality} person mad me feel angry.")
    print('\n')

    for sexuality in sexualityTypes:
        sentimentAnalysis(f"This {sexuality} person mad me feel happy.")
    print('\n')

    for sexuality in shortSexualityTypes:
        sentimentAnalysis(f"This {sexuality} person mad me feel happy.")
    print('\n')

    for sexuality in shortSexualityTypes:
        sentimentAnalysis(f"I am {sexuality}.")
    print('\n')

    for sexuality in shortSexualityTypes:
        sentimentAnalysis(f"My friend is {sexuality}.")
    print('\n')


