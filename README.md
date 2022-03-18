# Russian Endpoints
## Purpose
The goal of the Master's project was to search the Russian National Corpus (RNC) for to nested prepositional phrases to see whether particular patterns are present. Only a portion of the RNC is tagged for grammatical relationships the Deeply Annotated Corpus (DAC). In order to search the main corpus for grammatical relationships, I trained two Pytorch models on the DAC, one to identify the object of a particular preposition and one to identify the preposition that might be modifying a particular word. These models were used to process sentences from the main corpus to determine if they contained nesting prepositional phrases that could be used as evidence. 

## Linguistic Theory
In Russian, prepositional phrases indicating location and phrases indicating destination use the same set of prepositions, and are differentiated by the case of the object. If I wanted to say that I am 'in the university' the phrase would be 'в университете' [in university.LOC] but if I wanted to say that I am walking 'to the university' I would use the phrase 'в университет' [in university.ACC]. In this example the objects are in the locative and accusative cases respectively. In the case of nested locations such as "in the car, on the road," both phrases are usually either locational or directional, mostly depending on the verb. However a subsection of verbs, often describing a change of state, can take either locational or directional phrases and sometimes take as endpoints nested prepositional phrases that are mixed. For example:
  я      повесил     часы        на     окно        в комнате         Темпа. 
  Ya     povesil     chasy       na     okno        v komnatye        Tempa.
  I-NOM hung-PF   a clock-ACC   in.D a window-ACC in.L the room-LOC Tempe-Gen.
  I hung a clock on a window in Tempe's room. (RNC: Rid Grachev. In the 52nd year)


## How to Use

### Google Translation
During the project I used a Google translate API that required a Google Project and key to function. The key has expired, but if you set up your own Google project, you can insert your own API key. You would need to put the key in the folder and insert the name of the key into in RNCscript.py, main_network.py, and searchphraseconllu.py as os.environ["GOOGLE_APPLICATION_CREDENTIALS"].  
