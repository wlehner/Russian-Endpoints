# Russian Endpoints
## Purpose

The goal of the Master's project was to search the Russian National Corpus (RNC) for to nested prepositional phrases to see whether particular patterns are present. Only a portion of the RNC is tagged for grammatical relationships the Deeply Annotated Corpus (DAC). In order to search the main corpus for grammatical relationships, I trained two Pytorch models on the DAC, one to identify the object of a particular preposition and one to identify the preposition that might be modifying a particular word. These models were used to process sentences from the main corpus to determine if they contained nesting prepositional phrases that could be used as evidence. 

## Linguistic Theory
In Russian, prepositional phrases indicating location and phrases indicating destination use the same set of prepositions, and are differentiated by the case of the object. If I wanted to say that I am 'in the university' the phrase would be 'в университете' [in university.LOC] but if I wanted to say that I am walking 'to the university' I would use the phrase 'в университет' [in university.ACC]. In this example the objects are in the locative and accusative cases respectively. In the case of nested locations such as "in the car, on the road," both phrases are usually either locational or directional, mostly depending on the verb. However a subsection of verbs, typically describing a change of state, can take as endpoints either locational or directional phrases. As such these verbs can also take nested prepositional phrases that are mixed. For example:  
 - я      повесил     часы        на     окно        в комнате         Темпа.   
 - Ya     povesil     chasy       na     okno        v komnatye        Tempa.  
 - I-NOM hung-PF   a clock-ACC   in.D a window-ACC in.L the room-LOC Tempe-Gen.  
 - I hung a clock on a window in Tempe's room. (RNC: Rid Grachev. In the 52nd year)  
  
  As the example demonstrates, it is possible for the inner location to be directional and the outer locational to be locational. The purpose of this project was to determine if the reverse is possible as suggested by Blazhev[^1] or prohibited as suggested by Israeli [^2]. 

## How to Use
### Organization
- Corpora
  - Deeply Annotated Corpus: Stored in the three Conllu files ru_syntagrus-ud-dev.conllu, ru_syntagrus-ud-test.conllu, and ru_syntagrus-ud-train.conllu.
  - Main Corpus: Stored in sample_ar, within a series of folders. 
- Testing Results: The output of any testing was put into results_folder and each file contains all the relevant information about the test. Loss data for graphing is stored in the graphdata_folder and any graphs made were put into graphs. All of these have been numbered by test, such that scatterplot46.png, pointstoplot46, and result_layer2_46.txt all represent data from the same test.
- Trainscript.sh: This file was for scheduling jobs on the BU computing cluster.
- Training Models 
  - Main_network.py: Creates and trains the models.
  - Networks_folder: Contains all models, which are named by dimensions, whether they identify the object of a preposition (obj) or what a prepositional phrase is modifying (src).
  - mkvocablist.py: Searches the corpus and makes a word2vec dictionary of words and values. Word2vec assigns values based on words' similarity to one another and I used these as a way to translate sentences in Russian from the corpus into a form that can be used to train a NN. 
  - word2vec4.model: This is the word2vec model used to encode text into numbers that can be fed into the NN. 
- RNCscript.py: Searches the main corpus for possible examples using models.
- Searchphraseconllu.py: Searches the deeply annotated corpus for examples of nested prepositional phrases. 

### Google Translation
During the project I used a Google translate API that required a Google Project and key to function. I have removed my key, but if you set up your own Google project, you can insert your own API key. You would need to put the key in the folder and insert the name of the key into in RNCscript.py, main_network.py, and searchphraseconllu.py as os.environ["GOOGLE_APPLICATION_CREDENTIALS"].  

[^1] Blazhev, B. I. 1988. Upotreblenie Konstrukcii Napravlenija i Mesta v Sovremennom Russkom Jazyke. 2nd ed. Sofia: Narodna Prosveta.

[^2] Israeli, Alina. 2004. “Case Choice in Placement Verbs in Russian.” Glossos, no. 5: 1–54.
