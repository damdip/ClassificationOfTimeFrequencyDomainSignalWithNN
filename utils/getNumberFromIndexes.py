
##  dati degli indici come del tipo  A (1)  , questo restituisce il numero 1
##  dati degli indici come del tipo Z (2) , questo restituisce il numero 40* numero della lettera nell'alfabeto (A=0, B=1, ..., Z=25) + numero tra parentesi
##  dati degli indici come del tipo ZAA, ZAB, ..., ZAN , continuano la numerazione dopo Z

def getNumberFromIndexes(indexes: str) -> int:
    # Validazione formato base
    if '(' not in indexes or ')' not in indexes:
        raise ValueError("Formato invalido: mancano le parentesi")
    
    # Estrai il numero tra parentesi
    number_str = indexes[indexes.find('(')+1:indexes.find(')')]
    try:
        number = int(number_str)
    except ValueError:
        raise ValueError(f"Numero invalido tra parentesi: '{number_str}'")
    
    # Validazione range numero (1-40)
    if number < 1 or number > 40:
        raise ValueError(f"Numero deve essere tra 1 e 40, trovato: {number}")
    
    # Estrai la parte delle lettere (tutto prima della parentesi)
    letter_part = indexes[:indexes.find('(')].strip()
    
    # Gestisci i casi speciali ZAA-ZAN (dopo Z che va da 1000 a 1040)
    # ZAA inizia da 1041, ogni lettera successiva aggiunge 40
    special_cases = {
        'ZAA': 0, 'ZAB': 1, 'ZAC': 2, 'ZAD': 3, 'ZAE': 4, 
        'ZAF': 5, 'ZAG': 6, 'ZAH': 7, 'ZAI': 8, 'ZAJ': 9,
        'ZAK': 10, 'ZAL': 11, 'ZAM': 12, 'ZAN': 13
    }
    
    if letter_part in special_cases:
        # Z finisce a 1040 (25 * 40 + 40), quindi ZAA parte da 1040 + 1
        base = 26 * 40  # 1040
        return base + special_cases[letter_part] * 40 + number
    elif len(letter_part) == 1 and letter_part.isalpha() and letter_part.isupper():
        # Validazione lettera singola A-Z
        if letter_part < 'A' or letter_part > 'Z':
            raise ValueError(f"Lettera deve essere A-Z, trovata: {letter_part}")
        letter_value = ord(letter_part) - ord('A')
        return letter_value * 40 + number
    else:
        raise ValueError(f"Formato lettera invalido: '{letter_part}'. Ammessi solo A-Z o ZAA-ZAN")


# Test della funzione
if __name__ == "__main__":
    # Test casi normali A-Z
    print(f"asd (1) -> {getNumberFromIndexes('A (1)')}")  # Expected: 1
    print(f"A (40) -> {getNumberFromIndexes('A (40)')}")  # Expected: 40
    print(f"B (1) -> {getNumberFromIndexes('B (1)')}")  # Expected: 41
    print(f"Z (1) -> {getNumberFromIndexes('Z (1)')}")  # Expected: 1001
    print(f"Z (40) -> {getNumberFromIndexes('Z (41)')}")  # Expected: 1040
    
    # Test casi speciali ZAA-ZAN (continuano dopo Z (40) = 1040)
    print(f"ZAA (1) -> {getNumberFromIndexes('ZAA (1)')}")  # Expected: 1041
    print(f"ZAA (40) -> {getNumberFromIndexes('ZAA (40)')}")  # Expected: 1080
    print(f"ZAB (1) -> {getNumberFromIndexes('ZAB (1)')}")  # Expected: 1081
    print(f"ZAN (1) -> {getNumberFromIndexes('ZAN (1)')}")  # Expected: 1561
    
    # Test validazione
    print("\n--- Test validazione ---")
    try:
        getNumberFromIndexes('A (41)')
    except ValueError as e:
        print(f"✓ A (41) -> Errore: {e}")
    
    try:
        getNumberFromIndexes('A (0)')
    except ValueError as e:
        print(f"✓ A (0) -> Errore: {e}")
    
    try:
        getNumberFromIndexes('ZAO (1)')
    except ValueError as e:
        print(f"✓ ZAO (1) -> Errore: {e}")
    
    try:
        getNumberFromIndexes('a (1)')
    except ValueError as e:
        print(f"✓ a (1) -> Errore: {e}")
    
    print("\nTutti i test completati!")
