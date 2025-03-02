EXAMPLES = {
    "no_retrieve": [
        # [No Retrieval]
    """Question: Who was the first head of the Church of England?\n[No Retrieval]""",
    """Question: 'Roadhouse' is the ironic US Secret Service codename for what famous standard-setting luxury Art Deco hotel at 301 Park Avenue in Manhattan, New York City?\n[No Retrieval]""",
    """Question: Meaning literally 'make like' what is the full Latin word from which 'fax' derives (as in fax machine)?\n[No Retrieval]""",
    """Question: What does the Latin phrase ‘ab initio’ translate to in English?\n[No Retrieval]""",
    """Question: In which country is the town of Gorgonzola?\n[No Retrieval]""",
    """Question: Most of the Ozark Plateau is in which two US states?\n[No Retrieval]""",
    """Question: What was the name of the dog who accompanied the Three Men In A Boat?\n[No Retrieval]""",
    """Question: In poetry, a quatrain is a stanza or complete poem consisting of how many lines of verse?\n[No Retrieval]""",
    """Question: June 21, 1973, saw the US Supreme Court establish the Miller Test, which determines whether something is, or isnt, what?\n[No Retrieval]""",
    """Question: Who had a hit with 'What becomes of the broken hearted' in 1966 and again in 1974?\n[No Retrieval]""",

    ],

    "retrieve": [
        # [Retrieval]
    """Question: Who is the Chief Constable of the Greater Manchester Police Force?\n[Retrieval] Who is the Chief Constable of the Greater Manchester Police Force?""",
    """Question: "You get nothing for a pair" was a Bruce Forsyth catchphrase in which programme?\n[Retrieval] "You get nothing for a pair" was a Bruce Forsyth catchphrase in which programme?""",
    """Question: Who wrote How to Cheat at Cooking published in 1971?\n[Retrieval] Who wrote How to Cheat at Cooking published in 1971?""",
    """Question: Glassed-eyed member of the 'Rat Pack'?\n[Retrieval] Glassed-eyed member of the 'Rat Pack'?""",
    """Question: South African rugby winger Bryan Habana was named after which famous English sportsman?\n[Retrieval] South African rugby winger Bryan Habana was named after which famous English sportsman?""",
    """Question: Who went ‘Beyond Breaking Point’ in a Sport Relief challenge in March?\n[Retrieval] Who went ‘Beyond Breaking Point’ in a Sport Relief challenge in March?""",
    """Question: What is the surname of the character played by Joanna Lumley in 'Absolutely Fabulous'?\n[Retrieval] What is the surname of the character played by Joanna Lumley in 'Absolutely Fabulous'?""",
    """Question: What is the longest running show staged at London's Royal Drury Lane theatre?\n[Retrieval] What is the longest running show staged at London's Royal Drury Lane theatre?""",
    """Question: Bushido, developed between the 9th and 20th centuries relates to which culture?\n[Retrieval] Bushido, developed between the 9th and 20th centuries relates to which culture?""",

    """Question: 'Bloody Mary' refers to which queen of England?\n[Retrieval] 'Bloody Mary' refers to which queen of England?""",  # ['mary 1', 'Mary 1']
    """Question: Which service station is on the M6 toll motorway?\n[Retrieval] Which service station is on the M6 toll motorway?""",  # ['norton caines', 'Norton Caines']

    """Question: In the USA it's the Oscars what is it in France?\n[Retrieval] In the USA it's the Oscars what is it in France?""",  # ['caesars', 'caesar disambiguation', 'Caesar (disambiguation)', 'Casear', 'casear', 'Cesear', 'Caesare', 'Caesaros', 'Ceaser', 'caeser', 'caesaros', 'cæsar', 'Caesars', 'caesare', ...]

    """Question: Arnova, made by the French corporation Archos, is an 'entry level' brand of what?\n[Retrieval] Arnova, made by the French corporation Archos""",
    
    
    ],

    "planning": [
        # [Planning]
    # """Question: On the London underground only one station contains a single vowel. Which station?\n[Planning]""", # ['Banking system', '🏦', 'Banking business', 'Banking industry', '⛻', 'Credit institutions', 'Bank', 'Money-lenders', 'Banking establishment', 'banker', 'Credit Institutions', 'Monetary intermediation', 'credit institution', 'Banks and Banking', ...]
    # """Question: "Mr Tom Piperson" features in which of Beatrix Potter\'s stories?\n[Planning]""",
    # """Question: Zebra, Panda, Pelican, and Puffin are types of UK what?\n[Planning]""",  # ['pedestrian road crossings', 'Pedestrian road crossings']
    
    
    # """Question: Who had Wings like a shield of steel?\n[Planning]""", # ['batfink', 'The Short Circuit Case', 'Batfink: This Is Your Life', 'short circuit case', 'Pink Pearl of Persia', 'batfink this is your life', 'Batfink', 'pink pearl of persia']
    # from other dataset
    """Question: When was the director of film My Official Wife (1914 Film) born?\n[Planning]""",
    """Question: Which film whose director is younger, Men Without Law or Headlines (1925 Film)?\n[Planning]""",
    """Question: What is the place of birth of the director of film Martha, Meet Frank, Daniel And Laurence?\n[Planning]""",
    """Question: Who was the lead singer of the manhattans?\n[Planning]""",
    """Question: What is the name of the matchmaker in fiddler?\n[Planning]""",
    """Question: what is your dragon from how to train your dragon\n[Planning]""",
    """Question: Who was the composer of Two?\n[Planning]""",
    """Question: What genre is Foxy Brown?\n[Planning]""",  # Pete Townshend

    ]
}
