import spacy
import contextualSpellCheck

nlp = spacy.load('en_core_web_lg')
contextualSpellCheck.add_to_pipe(nlp)

original_sentence = 'Income was $9.4 milion compared to the prior year of $2.7 milion.'
doc = nlp(original_sentence)

print("Original sentence:")
print(original_sentence)

print("Performed spell check:")
print(doc._.performed_spellCheck) #Should be True

print ("Corrected to:")
print(doc._.outcome_spellCheck) #Income was $9.4 million compared to the prior year of $2.7 million.
