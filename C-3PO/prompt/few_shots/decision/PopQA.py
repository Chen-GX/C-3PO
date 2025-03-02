EXAMPLES = {
    "no_retrieve": [
        # [No Retrieval]
    """Question: Who was the producer of Roy?\n[No Retrieval]""",
    """Question: What is the capital of Curaçao?\n[No Retrieval]""",
    """Question: What is the capital of County Kilkenny?\n[No Retrieval]""",
    """Question: Who was the screenwriter for These Three?\n[No Retrieval]""",
    """Question: Who is the author of Fihi Ma Fihi?\n[No Retrieval]""",
    """Question: In what country is Port Authority Trans-Hudson?\n[No Retrieval]""",
    """Question: What is the religion of Rama?\n[No Retrieval]""",
    """Question: What sport does Rugby Africa play?\n[No Retrieval]""",
    """Question: Who is the father of Anil Kapoor?\n[No Retrieval]""",
    """Question: Who was the director of Sugar?\n[No Retrieval]""",

    ],

    "retrieve": [
        # [Retrieval]
    """Question: Who was the director of Skull Heads?\n[Retrieval] Who was the director of Skull Heads?""",
    """Question: What genre is 454 Big Block?\n[Retrieval] What genre is 454 Big Block?""",
    """Question: Who was the screenwriter for The Sea Inside?\n[Retrieval] Who was the screenwriter for The Sea Inside?""",
    """Question: Who was the screenwriter for Greed?\n[Retrieval] Who was the screenwriter for Greed?""",  # ['William Graham Sumner']
    """Question: In what city was Jack Kachkar born?\n[Retrieval] In what city was Jack Kachkar born?""",
    """Question: What genre is World Trade?\n[Retrieval] What genre is World Trade?""",  # ['26 July 2017']
    """Question: What is Sarai the capital of?\n[Retrieval] What is Sarai the capital of?""",  # Gregory Abbott
    """Question: Who is the author of Voyage d'Egypte et de Nubie?\n[Retrieval] Voyage d'Egypte et de Nubie""",  # 必须做filter
    # """Question: Who was the producer of La Mission?\n[Retrieval] Who was the producer of La Mission?""",  # 必须做filter
    # """Question: What sport does Paulo Grilo play?\n[Retrieval]""",  # ['association football', 'football', 'soccer']
    """Question: In what city was Barbara Harris born?\n[Retrieval] Barbara Harris""",  # ['Philadelphia', 'Philly', 'City of Brotherly Love', 'Cradle of Liberty', 'Philadelphia, Pennsylvania', 'City of Philadelphia', 'Philadelphia, PA']
    """Question: What is Jacopo Melani's occupation?\n[Retrieval] Jacopo Melani""",
    """Question: What is the religion of Juan Soldevilla y Romero?\n[Retrieval] What is the religion of Juan Soldevilla y Romero?""",
    """Question: What is Laishevo the capital of?\n[Retrieval] What is Laishevo the capital of?""",
    
    
    ],

    "planning": [
        # [Planning]
    """Question: Who was the composer of Two?\n[Planning]""",
    """Question: What genre is Foxy Brown?\n[Planning]""",  # Pete Townshend
    """Question: Who was the producer of The Piano?\n[Planning]""",  # Robert Southey
    # """Question: What is the religion of Francis?\n[Planning]""",
    # """Question: Who is the author of The Search?\n[Planning]""",
    # """Question: Who was the screenwriter for The Bench?\n[Planning]""",  # 检索不出来
    """Question: Who was the director of The Wedding?\n[Planning]""",
    ]
}
