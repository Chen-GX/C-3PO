EXAMPLES = {
    "no_retrieve": [
        # [No Retrieval]
    """Question: who said a plague on both your houses\n[No Retrieval]""",
    """Question: when did the song mr. sandman come out\n[No Retrieval]""",
    """Question: when did ringling brothers merge with barnum and bailey\n[No Retrieval]""",
    """Question: what were the two opposing sides in china 's civil war\n[No Retrieval]""",
    """Question: when did pawn stars first air on tv\n[No Retrieval]""",
    """Question: who is the original singer of how sweet it is to be loved by you\n[No Retrieval]""",
    """Question: where did they film transformers age of extinction\n[No Retrieval]""",
    """Question: the portion of the uterine endometrium that is shed every month is the\n[No Retrieval]""",
    """Question: once upon a time in mumbaai based on whose story\n[No Retrieval]""",

    ],

    "retrieve": [
        # [Retrieval]
    """Question: who died in the beginning of fast and furious\n[Retrieval] who died in the beginning of fast and furious""",
    """Question: who appeared on saturday night live when adele was the musical guest in 2008\n[Retrieval] who appeared on saturday night live when adele was the musical guest in 2008""",
    """Question: who has the most nba championships in the nba\n[Retrieval] who has the most nba championships in the nba""",
    """Question: the american academician who taught the first sociology courses in the united states was\n[Retrieval] the american academician who taught the first sociology courses in the united states was""",  # ['William Graham Sumner']
    """Question: where does cape towns water supply come from\n[Retrieval] where does cape towns water supply come from""",
    """Question: when does 47 meters down come out in uk\n[Retrieval] when does 47 meters down come out in uk""",  # ['26 July 2017']
    """Question: who sang i want to shake you down\n[Retrieval] who sang i want to shake you down""",  # Gregory Abbott
    """Question: when was sound captured for the first time\n[Retrieval] when was sound captured for the first time""",  # 必须做filter
    """Question: where does the last name barnes come from\n[Retrieval] where does the last name barnes come from""",  # 必须做filter
    
    
    ],

    "planning": [
        # [Planning]
    """Question: what is your dragon from how to train your dragon\n[Planning]""",

    # from other dataset
    """Question: When was the director of film My Official Wife (1914 Film) born?\n[Planning]""",
    """Question: Which film whose director is younger, Men Without Law or Headlines (1925 Film)?\n[Planning]""",
    """Question: What is the place of birth of the director of film Martha, Meet Frank, Daniel And Laurence?\n[Planning]""",
    """Question: Who was the lead singer of the manhattans?\n[Planning]""",
    """Question: What is the name of the matchmaker in fiddler?\n[Planning]""",
    """Question: Who was the composer of Two?\n[Planning]""",
    """Question: What genre is Foxy Brown?\n[Planning]""",  # Pete Townshend

    ]
}
