# Scoring Sanity Check

Review these 50 examples to ensure normalization and alias matching are working correctly.

## Example 1
**Question:** Which theory states that 'people tend to rise to their own level of incompetence'?

**Gold Aliases:** ['Hierarchiology', 'Promote to fail', 'Peter Principal', 'Tabulology', "Peter's Principle", 'Peter principal', 'PETER PRINCIPLE', 'The Peter Principle', 'Managing upward', 'Peter principle', 'Peter Principle']

**Computed k (Score):** 4/10

**Generations:**
- [❌] Answer: Maslow's Hierarchy of Needs theory states that 'people tend to rise to their own level of incompetence'.
- [✅] Answer: The Peter Principle theory.
- [❌] Answer: The Law of the Incompetent.
- [❌] Answer: The Law of the Incompetent.
- [✅] Answer: The Peter Principle.
- [❌] Answer: Maslow's Hierarchy of Needs theory does not state this.
- [✅] Answer: The Peter Principle theory.
- [❌] Answer: The Law of the Minimum.
- [✅] Answer: The Peter Principle.
- [❌] Answer: Abilene Paradox and Dunning-Kruger effect.

---

## Example 2
**Question:** Which American actor also performs with the band Dogstar?

**Gold Aliases:** ['Keanu Reaves', 'Ava Archer Syme-Reeves', 'Keanu', 'Keanu reeves', 'Filmography of Keanu Reeves', 'Keanu Reeves', 'Keanau Reeves', 'Sad Keanu', 'Sad keanu', 'Keanu Reeves/Filmography', 'Keanu Reevs', 'Kenau Reeves', 'Keanu Reeves filmography', 'Kenu reeves', 'Cool breeze over the mountains']

**Computed k (Score):** 0/10

**Generations:**
- [❌] Answer: Chris Potter
- [❌] Answer: Jason Mraz
- [❌] Answer: Mark Mothersbaugh
- [❌] [FALLBACK] Robbie Finn
- [❌] Answer: Chris Isaak.
- [❌] Answer: David Spade
- [❌] Answer: Devon Sawa
- [❌] Answer: Jason Alexander
- [❌] Answer: Wil Wheaton
- [❌] Answer: Scott Caan

---

## Example 3
**Question:** Which writer was the father of Christopher Robin?

**Gold Aliases:** ['A a milne', 'A A MILNE', 'A. A. Milne', 'A Milne', 'Dorothy Milne', 'Alan Alexander Milne', 'AA Milne', 'Alan Milne', 'Daphne de Sélincourt', 'A A Milne', 'A.A. Milne', 'Daphne Milne', 'Dorothy de Sélincourt']

**Computed k (Score):** 10/10

**Generations:**
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] Answer: A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne
- [✅] [FALLBACK] A.A. Milne

---

## Example 4
**Question:** The Dance Class and Absinthe are famous works by which painter?

**Gold Aliases:** ['Hilaire-Germain-Edgar Degas', 'Hilaire Germain Edgar Degas', 'Degas', 'Edgar Dégas', 'Edgar De Gas', 'Edgar Degas', 'EDGAR DEGAS', 'Degas, Hilaire Germain Edgar', 'De Gas']

**Computed k (Score):** 3/10

**Generations:**
- [❌] Answer: Edvard Munch
- [✅] Answer: Edgar Degas
- [❌] [FALLBACK] Edvard Munch
- [❌] Answer: Henri de Toulouse-Lautrec
- [✅] Answer: Edgar Degas.
- [❌] Answer: Edvard Munch
- [❌] [FALLBACK] Edouard Manet
- [❌] Answer: Henri de Toulouse-Lautrec
- [❌] [FALLBACK] Claude Monet
- [✅] Answer: Edgar Degas

---

## Example 5
**Question:** In which city was the British version of Concorde assembled

**Gold Aliases:** ['Bristol (England)', 'Bristol, United Kingdom', 'Bristol', 'Bristol, UK', 'Bristol, England', 'Bristolshire', 'City of Bristol, United Kingdom', 'Bristol; City of UA', 'Brycgstow', 'County Borough of Bristol', 'Bristol (district)', 'Westbury Park Primary School', 'UN/LOCODE:GBBRS', 'BRISTOL', 'Brizzle', 'The weather in Bristol', 'City of Bristol', 'Unitary Authority of Bristol']

**Computed k (Score):** 2/10

**Generations:**
- [❌] [FALLBACK] Filton
- [❌] [FALLBACK] London
- [❌] [FALLBACK] Filton
- [✅] [FALLBACK] Filton, Bristol.
- [❌] [FALLBACK] Filton
- [❌] [FALLBACK] Filton, Gloucestershire
- [❌] [FALLBACK] Filton
- [✅] [FALLBACK] Filton, near Bristol.
- [❌] [FALLBACK] Filton.
- [❌] [FALLBACK] Buckingham

---

