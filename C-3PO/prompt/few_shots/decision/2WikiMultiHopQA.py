# decision_prompt = {
#     "no_retrieve": [
#         # [No Retrieval]
#     """Question: Are the directors of films Esterina and Fence Riders both from the same country?\n[No Retrieval]""",
#     """Question: Which award the performer of song Proud Mary earned?\n[No Retrieval]""",
#     """Question: Where was the director of film Lost In Siberia born?\n[No Retrieval]""",
#     ],
#     "retrieve": [
#         # [Retrieval]
#     """Question: Are both bands, Chris Robinson Brotherhood and Outbreakband, from the same country?\n[Retrieval] Are Chris Robinson Brotherhood and Outbreakband from the same country?""",
#     """Question: Are both The Willowz and Deepfield from the same country?\n[Retrieval] Are both The Willowz and Deepfield from the same country?""",
#     # """Question: when was the first general election held in india?\n[Retrieval] when was the first general election held in india?""",
#     # """Question: Qbasic is the extension of which programming language?\n[Retrieval] Qbasic is the extension of which programming language?""",
#     ],
#     "planning": [
#         # [Planning]
#     """Question: What nationality is the director of film The Caper Of The Golden Bulls?\n[Planning] To answer the question, we need to find information about the director of the film "The Caper of the Golden Bulls." Then we should determine which nationality is the director born using the retrieval.\nStep 1: Find the relevant documents that mention the film `The Caper of the Golden Bulls.`\nStep 2: Identify the director of the film from the retrieved documents.\nStep 3: Find the relevant information about `Which nationality is the director born?`.\nStep 4: Provide the answer based on the retrieved information.""",
#     """Question: When is Dominic Roco's father's birthday?\n[Planning] I don't have access to personal information about who is the father of Dominic Roco, including his birthday. However, we can retrieve information to answer the question about Dominic Roco's father's birthday.\nSTEP 1: We should find the relevant documents about `Dominic Roco` to identify his father's name.\nSTEP 2: Identify the father's name from the retrieved documents.\nSTEP 3: Find the relevant information about the father's birthday.\nSTEP 4: Provide the answer based on the retrieved information.""",
#     """Question: Why did the director of film The Notorious Landlady die?\n[Planning] To answer the question, we need to find information about `the director of film The Notorious Landlady`. Then, we should retrieve the cause of the director's death.\nSTEP 1: Retrieve the relevant documents about `the director of film The Notorious Landlady`.\nSTEP 2: Identify the director of the film from the retrieved documents.\nSTEP 3: Find the relevant information about `Why did the director die?`.\nSTEP 4: Finish the answer based on the retrieved information.""",
#     """Question: Where was the place of death of the director of film Magic Mirror (Film)?\n[Planning] We know that the film "Magic Mirror" was directed by Manoel de Oliveira. To answer the question, we need to find information about the place of death of Manoel de Oliveira.\nSTEP 1: Retrieve the relevant documents about `Where was the place of death of Manoel de Oliveira?`.\nSTEP 2: Find the place of death of Manoel de Oliveira from the retrieved documents and provide the answer.""",
#     ]
# }



EXAMPLES = {
    "no_retrieve": [
        # [No Retrieval]
    """Question: Do both films The Falcon (Film) and Valentin The Good have the directors from the same country?\n[No Retrieval]""",
    """Question: Where did the director of film Ride The Man Down die?\n[No Retrieval]""",
    """Question: Which film has the director born later, The Countess Of Parma or Prem Bandhan?\n[No Retrieval]""",
    """Question: Did the movies Karılar Koğuşu and A Pizza In Jordbro, originate from the same country?\n[No Retrieval]""",
    """Question: What is the place of birth of the composer of song Gretchen Am Spinnrade?\n[No Retrieval]""",
    """Question: Were Mary Schiavo and Faisal Al-Dakhil from the same country?\n[No Retrieval]""",
    """Question: Are the movies The Market Of Vain Desire and Asokamala, from the same country?\n[No Retrieval]""",
    ],
    "retrieve": [
        # [Retrieval]
    """Question: Who is the stepchild of Lysicles (5Th Century Bc)?\n[Retrieval] Who is the stepchild of Lysicles (5Th Century Bc)?""",
    """Question: Who is the father-in-law of Infanta Blanca Of Spain?\n[Retrieval] Who is the father-in-law of Infanta Blanca Of Spain?""",
    """Question: Which film was released first, Sweet And Twenty or Caravan Of Death (Film)?\n[Retrieval] Which film was released first, Sweet And Twenty or Caravan Of Death (Film)?""",
    """Question: What is the date of birth of E. C. Spykman's husband?\n[Retrieval] What is the date of birth of E. C. Spykman's husband?""",
    """Question: Why did the performer of song Someday My Day Will Come die?\n[Retrieval] Why did the performer of song Someday My Day Will Come die?""",
    """Question: Do director of film The Nines and director of film The Sea Wolf (1920 Film) share the same nationality?\n[Retrieval] Do director of film The Nines and director of film The Sea Wolf (1920 Film) share the same nationality?""",
    """Question: Are the directors of films Joe (2013 Film) and Boynton Beach Club both from the same country?\n[Retrieval] Are the directors of films Joe (2013 Film) and Boynton Beach Club both from the same country?""",
    """Question: Which film has the director born earlier, Raja Kumarudu or Into The Abyss (Film)?\n[Retrieval] Which film has the director born earlier, Raja Kumarudu or Into The Abyss (Film)?""",
    """Question: What is the place of birth of the director of film The Road To Denver?\n[Retrieval] What is the place of birth of the director of film The Road To Denver?""",
    """Question: What is the date of death of Charles-René D'Hozier's father?\n[Retrieval] What is the date of death of Charles-René D'Hozier's father?""",

    ],
    "planning": [
        # [Planning]
    """Question: When was the director of film My Official Wife (1914 Film) born?\n[Planning]""",
    """Question: Which film whose director is younger, Men Without Law or Headlines (1925 Film)?\n[Planning]""",
    """Question: What is the place of birth of the director of film Martha, Meet Frank, Daniel And Laurence?\n[Planning]""",
    """Question: Where was the father of Mirjam Finkelstein born?\n[Planning]""",
    """Question: What is the date of death of Joan Of Dampierre's mother?\n[Planning]""",
    """Question: Which film whose director is younger, The Vein or Judgment Deferred?\n[Planning]""",
    """Question: What is the date of death of the director of film Little Man, What Now? (1933 Film)?\n[Planning]""",
    """Question: Which film was released first, Welcome To Home Gori or Good Sam?\n[Planning]""",
    """Question: Where did the director of film Happy Ghost Iv study?\n[Planning]""",
    """Question: What is the date of death of the director of film The Sporting Lover?\n[Planning]""",
    """Question: Who is Prince Moulay Rachid Of Morocco's paternal grandmother?\n[Planning]""",
    """Question: Why did the director of film Being Respectable die?\n[Planning]""",
    """Question: Are both Scatterwood Lake and Hazeltine Lake located in the same country?\n[Planning]""",
    """Question: When is the director of film Mickey One 's birthday?\n[Planning]""",
    """Question: Are both Liege/Cnrl Aerodrome and Deer Lake Airport located in the same country?\n[Planning]""",
    """Question: When did the director of film Mutthu Ondu Mutthu die?\n[Planning]""",
    """Question: Where did the director of film Two Tickets To Broadway graduate from?\n[Planning]""",
    """Question: Who is the paternal grandfather of John Iv, Count Of Soissons?\n[Planning]""",
    ]
}