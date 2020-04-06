from utils.loader import *
from utils.book_keeping import *


class Lexicon():
    def __init__(self, language):
        self.language = language
        self.lexicon = load_lexicon(self.language)

        self.features = Features[self.language]
        
    def lookup(self, word):
        results = self.lexicon[self.lexicon["Word"] == word]
        return(results)

    def values_for_word(self, word, feature):
        """
        This tests how many possible feature values of a given feature
        a particular word can take. This is used to see how ambiguous
        a word is with respect to the feature. 
        For example, `eat' in English is three-way ambiguous for Person
        I eat (1st), you eat (2nd), they eat (3rd)
        """
        possible_values = [value for value in self.features[feature]]
        attributed_values = []
        results = self.lookup(word)
        for _, row in results.iterrows():
            try:
                tag = [item for item in row["Features"].split("|") if feature in item][0]
                for value in possible_values:
                    if value in tag and value not in attributed_values:
                        attributed_values.append(value)
            except:
                continue            
        return(attributed_values)
                    
            

    
