from imports import *

## Lowercase, trim, and remove non-letter characters
def normalizeString(string):
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string

def getData(data_path, pairs_path):
    qa_pairs = []
    sentences_length = []
    with open(data_path, "r", encoding='utf8') as f:
        # Load the data
        for line in f:
            sentences = line.split("__eou__")
            sentences.pop()                                               # Remove the end-line
            sentences_length.append(len(sentences))                       # Save the length of the conversation
            # Separate conversation into pairs of sentences

            if path.exists(pairs_path) == False or os.stat(pairs_path).st_size == 0:  # if the trimmed pairs are already saved
                for idx in range(len(sentences)-1):
                    inputLine = normalizeString(sentences[idx].strip())           # .strip() removes start/end whitespace
                    targetLine = normalizeString(sentences[idx+1].strip())
                    if inputLine and targetLine:
                        qa_pairs.append([inputLine, targetLine])

    # Write pairs into csv file
    if path.exists(pairs_path) == False or os.stat(pairs_path).st_size == 0:          # if the trimmed pairs are already saved
        delimiter = '\t'
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))
        with open(pairs_path, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter = delimiter, lineterminator='\n')
            for pair in qa_pairs:
                writer.writerow(pair)

    return sentences_length      


# Keep the first line from each conversation
# The lines serve as first sentence (input) for the chatbot conversation
def getTestData(data_path, lines_path, max_length):
    lines = []
    with open(data_path, "r") as f:
        # Load data
        for line in f:
            sentences = line.split("__eou__")
            sentences.pop()
            inputLine = normalizeString(sentences[0].strip())

            if inputLine and len(inputLine.split(" ")) < max_length:
                lines.append(inputLine)            

    with open(lines_path, 'w', encoding='utf-8') as outputfile:
        for line in lines:
            outputfile.write(line)
            outputfile.write('\n')
        outputfile.close()        