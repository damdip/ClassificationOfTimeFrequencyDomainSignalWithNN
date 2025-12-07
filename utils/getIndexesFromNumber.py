## Funzione inversa: dato un numero compreso tra 1 e 1600, restituisce l'indice del file
## Es: 1 -> "A (1)", 40 -> "A (40)", 41 -> "B (1)", 1040 -> "Z (40)"
## 1041 -> "ZAA (1)", 1600 -> "ZAN (40)"

def getIndexesFromNumber(number: int) -> str:
    # Validazione range numero (1-1600)
    if number < 1 or number > 1600:
        raise ValueError(f"Numero deve essere tra 1 e 1600, trovato: {number}")
    
    # Calcola la lettera e il numero tra parentesi
    # Ogni lettera copre 40 numeri (da 1 a 40)
    letter_index = (number - 1) // 40  # 0-based index della lettera
    file_number = ((number - 1) % 40) + 1  # Numero da 1 a 40
    
    # Gestisci i casi speciali ZAA-ZAN (letter_index >= 26)
    special_cases = ['ZAA', 'ZAB', 'ZAC', 'ZAD', 'ZAE', 'ZAF', 'ZAG', 
                     'ZAH', 'ZAI', 'ZAJ', 'ZAK', 'ZAL', 'ZAM', 'ZAN']
    
    if letter_index >= 26:
        # ZAA inizia da letter_index 26
        special_index = letter_index - 26
        if special_index < len(special_cases):
            letter_part = special_cases[special_index]
        else:
            raise ValueError(f"Numero {number} fuori range per casi speciali")
    else:
        # Lettere normali A-Z (letter_index 0-25)
        letter_part = chr(ord('A') + letter_index)
    
    return f"{letter_part} ({file_number})"


# Test della funzione
if __name__ == "__main__":
    # Test casi normali A-Z
    print(f"1 -> {getIndexesFromNumber(1)}")  # Expected: A (1)
    print(f"40 -> {getIndexesFromNumber(40)}")  # Expected: A (40)
    print(f"41 -> {getIndexesFromNumber(41)}")  # Expected: B (1)
    print(f"1001 -> {getIndexesFromNumber(1001)}")  # Expected: Z (1)
    print(f"1040 -> {getIndexesFromNumber(1040)}")  # Expected: Z (40)
    
    # Test casi speciali ZAA-ZAN
    print(f"1041 -> {getIndexesFromNumber(1041)}")  # Expected: ZAA (1)
    print(f"1080 -> {getIndexesFromNumber(1080)}")  # Expected: ZAA (40)
    print(f"1081 -> {getIndexesFromNumber(1081)}")  # Expected: ZAB (1)
    print(f"1561 -> {getIndexesFromNumber(1561)}")  # Expected: ZAN (1)
    print(f"1600 -> {getIndexesFromNumber(1600)}")  # Expected: ZAN (40)
    
    # Test validazione
    print("\n--- Test validazione ---")
    try:
        getIndexesFromNumber(0)
    except ValueError as e:
        print(f"✓ 0 -> Errore: {e}")
    
    try:
        getIndexesFromNumber(1601)
    except ValueError as e:
        print(f"✓ 1601 -> Errore: {e}")
    
    # Test simmetria con getNumberFromIndexes
    print("\n--- Test simmetria ---")
    from getNumberFromIndexes import getNumberFromIndexes
    for num in [1, 40, 41, 1040, 1041, 1600]:
        indexes = getIndexesFromNumber(num)
        back = getNumberFromIndexes(indexes)
        print(f"{num} -> {indexes} -> {back} {'✓' if num == back else '✗'}")
    
    print("\nTutti i test completati!")
