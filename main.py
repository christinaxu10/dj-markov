import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Global vars
notes = ["A", "B", "C", "D", "E", "F", "G"]
accs = ["b", "s", ""]
all_notes_list = [note + acc for note in notes for acc in accs]

keys_list = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

### Data cleaning ###
def simplify_chord(chord: str) -> str:
    """
    Removes chord quality from a chord.
    """
    for note in all_notes_list:
        if not chord.startswith(note):
            continue

        suffix = chord.removeprefix(note)
        if suffix.startswith("min") or suffix.startswith("dim"):
            return note + "min"
        else:
            return note

    if chord == "sC":
        return "Cs"

    # print(chord)
    return ""


### Transposing each song to every other key ###
# For data augmentation & mitigating bias from song key
def transpose_chord(chord: str, variation: int) -> str:
    # Find the base note and suffix
    for note in keys_list:
        if chord.startswith(note):
            suffix = chord.removeprefix(note)
            idx = keys_list.index(note)
            new_note = keys_list[(idx + variation) % 12]
            return new_note + suffix
    return chord  # If not found, return as is

def augment_keys(df):
    augmented_rows = []
    for _, row in df.iterrows():
        for variation in range(12):
            new_row = row.copy()
            if variation == 0:
                new_row["original_key"] = True
            else:
                new_row["original_key"] = False
            new_row["added_semitones"] = variation
            new_row["chords"] = [transpose_chord(chord, variation) for chord in row["chords"]]
            augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)


### Calculating n-gram counts ###
def load_chord_data(series) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    unigram_dict: dict[str, int] = {}
    bigram_dict: dict[str, dict[str, int]] = {}

    for row in series:
        # Unigram counts
        for chord in row:
            unigram_dict[chord] = unigram_dict.get(chord, 0) + 1
        # Bigram counts
        for i in range(len(row) - 1):
            w1, w2 = row[i], row[i + 1]
            if w1 not in bigram_dict:
                bigram_dict[w1] = {}
            bigram_dict[w1][w2] = bigram_dict[w1].get(w2, 0) + 1

    return unigram_dict, bigram_dict

def count_n_grams(data, n: int = 1) -> pd.DataFrame:
    word_vectorizer = CountVectorizer(
        ngram_range=(1, n),
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )

    sparse_matrix = word_vectorizer.fit_transform(
        data.map(lambda chords: " ".join(chords))
    )

    frequencies = sum(sparse_matrix).toarray()[0]

    df_all = pd.DataFrame(
        frequencies,
        index=word_vectorizer.get_feature_names_out(),
        columns=["count"],
    )

    return df_all.groupby(by=lambda chords: len(chords.split(" ")))


### Song log likelihood calculation ###
def song_log_likelihood_ngram(song, n, ngram_probs):
    # song: list of chords in song
    # n: order of the n-gram model
    # ngram_probs: dict[context_tuple] -> dict[target] = P(target | context)
    # ex: trigram ngram_prob = dict[(chord1, chord2)] = {chord0:P,...,chordV:P}, dict[chord3] = P(chord3 | chord1, chord2)
    # vocab_size: 42 or 36?

    ll = 0.0
    if len(song) < n:
        return 0.0

    for t in range(n-1, len(song)):
        context = tuple(song[t-(n-1):t])
        target = song[t]

        if context in ngram_probs:
            p = ngram_probs[context].get(target, 0.0)
        else:
            p = 1e-12

        if p <= 0:
            p = 1e-12
        ll += np.log(p)

    return ll


def main():
    # df = pd.read_csv("hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv",usecols=["chords", "main_genre"])

    # pop_chords = df[df["main_genre"] == "pop"]["chords"]
    # pop_chords = pop_chords.str.split(" ")
    # pop_chords = pop_chords.map(
    #     lambda chords: [simplify_chord(chord) for chord in chords if not chord.startswith("<")]
    # )
    # pop_chords["original_key"] = True
    # pop_chords["added_semitones"] = 0

    # # Transpose songs
    # pop_chords = augment_keys(pop_chords)

    pop_chords = pd.read_csv('chordonomicon_v2_augmented.csv') # transposed pop songs

    n = 2
    alpha = 1.0 # Laplace smoothing

    # Get counts for all n-grams
    n_gram_counts = count_n_grams(pop_chords, n)
    # print(n_gram_counts)
    for key, item in n_gram_counts:
        print(n_gram_counts.get_group(key).sort_values(by='count'), "\n\n")
    
    observed_chords = sorted(pop_chords.map(set).agg(lambda x: set.union(*x))) # vocubulary
    print(observed_chords)

    # n_gram_counts = n_gram_counts.groupby(lambda x: len(x.split(" ")))
    
    # Calculate transition matrix probabilities
    unigram = n_gram_counts.get_group(1)
    unigram = unigram.reindex(all_notes_list, fill_value=0)
    unigram["prob"] = (unigram["count"] + alpha) / (unigram["count"].sum() + alpha * len(observed_chords))

    bigram = n_gram_counts.get_group(2)
    bigram["evidence"] = bigram.index.map(lambda s: s.split()[0]) # get (n-1)-length evidence
    bigram["next"] = bigram.index.map(lambda s: s.split()[1]) # next chords

    full_index = pd.MultiIndex.from_product([all_notes_list, all_notes_list], names=["evidence", "next"])
    bigram = bigram.set_index(["evidence", "next"])
    bigram = bigram.reindex(full_index, fill_value=0)

    evidence_counts = bigram["count"].groupby(level="evidence").transform("sum")
    bigram["prob"] = (bigram["count"] + alpha) / (evidence_counts + alpha * len(bigram["next"].unique()))

    # 2d dataframe
    transition_matrix = bigram["prob"].unstack(fill_value=0.0)


if __name__ == "__main__":
    main()
