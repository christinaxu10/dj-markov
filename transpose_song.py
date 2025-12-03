import pandas as pd

notes = ["A", "B", "C", "D", "E", "F", "G"]
accs = ["b", "s", ""]
all_notes_list = [note + acc for note in notes for acc in accs]

keys_list = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

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

    print(chord)
    return ""

def transpose_chord(chord: str, variation: int) -> str:
    # Identify base note and suffix robustly
    if len(chord) >= 2 and chord[1] == "s":
        base = chord[:2]
        suffix = chord[2:]
    else:
        base = chord[:1]
        suffix = chord[1:]
    if base in keys_list:
        idx = keys_list.index(base)
        new_base = keys_list[(idx + variation) % 12]
        return new_base + suffix
    return chord  # If not found, return as is

def standardize_chord_prefix(chord: str) -> str:
    # Map flat notes to their sharp equivalents
    flat_to_sharp = {
        "Bb": "As",
        "Db": "Cs",
        "Eb": "Ds",
        "Gb": "Fs",
        "Ab": "Gs",
        "Bs": "C",
        "Es": "F"
    }
    for flat, sharp in flat_to_sharp.items():
        if chord.startswith(flat):
            return sharp + chord[len(flat):]
    return chord

def get_pop_chords_df():
    df = pd.read_csv("chordonomicon_v2.csv", usecols=["id", "chords", "main_genre"])
    pop_df = df[df["main_genre"] == "pop"][["id", "chords"]].copy()
    pop_df["chords"] = pop_df["chords"].str.split(" ")
    pop_df["chords"] = pop_df["chords"].map(
        lambda chords: [chord for chord in chords if not chord.startswith("<")]
    )
    pop_df["chords"] = pop_df["chords"].map(
        lambda chords: [standardize_chord_prefix(chord) for chord in chords]
    )
    pop_df["chords"] = pop_df["chords"].map(
        lambda chords: [simplify_chord(chord) for chord in chords]
    )
    pop_df["original_key"] = True
    pop_df["added_semitones"] = 0
    return pop_df

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

# Example usage
if __name__ == "__main__":
    pop_chords_df = get_pop_chords_df()
    augmented_df = augment_keys(pop_chords_df)
    print(augmented_df.head(15))
    augmented_df.to_csv("chordonomicon_v2_augmented.csv", index=False)