from imports import *

## Lowercase, trim, and remove non-letter characters
def normalizeString(string):
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

def getData(DATA_PATH, pairs_trimmed):
    qa_pairs = []
    sentences_length = []
    with open(DATA_PATH, "r") as f:
        # Load the data
        for line in f:
            sentences = line.split("__eou__")
            sentences.pop()                                               # Remove the end-line
            sentences_length.append(len(sentences))
            # Separate conversation into pairs of sentences
            for idx in range(len(sentences)-1):
                inputLine = normalizeString(sentences[idx].strip())       # .strip() removes start/end whitespace
                targetLine = normalizeString(sentences[idx+1].strip())
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])

    # Write pairs into csv file
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    with open(pairs_trimmed, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter = delimiter, lineterminator='\n')
        for pair in qa_pairs:
            writer.writerow(pair)

    return sentences_length        