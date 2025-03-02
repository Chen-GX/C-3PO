EXAMPLES = [
    """Question: What nationality is the director of film The Caper Of The Golden Bulls?\nTo answer the question, we need to find information about the director of the film "The Caper of the Golden Bulls." Then we should determine which nationality is the director born using the retrieval.\nStep 1: Retrieve the relevant documents that mention the film `The Caper of the Golden Bulls.`\nStep 2: Identify the director of the film from the retrieved documents.\nStep 3: Retrieve the relevant information about `Which nationality is the director born?`.\nStep 4: Provide the answer based on the retrieved information.""",
    """Question: When is Dominic Roco's father's birthday?\nI don't have access to personal information about who is the father of Dominic Roco, including his birthday. However, we can retrieve information to answer the question about Dominic Roco's father's birthday.\nSTEP 1: We should retrieve the relevant documents about `Dominic Roco` to identify his father's name.\nSTEP 2: Retrieve the relevant information about the father's birthday.\nSTEP 3: Provide the answer based on the retrieved information.""",
    """Question: Why did the director of film The Notorious Landlady die?\nTo answer the question, we need to find information about `the director of film The Notorious Landlady`. Then, we should retrieve the cause of the director's death.\nSTEP 1: Retrieve the relevant documents about `the director of film The Notorious Landlady`.\nSTEP 2: Identify the director of the film from the retrieved documents.\nSTEP 3: Retrieve the relevant information about `Why did the director die?`.\nSTEP 4: Finish the answer based on the retrieved information.""",

    """Question: Where was the place of death of the director of film Magic Mirror (Film)?\nWe know that the film "Magic Mirror" was directed by Manoel de Oliveira. To answer the question, we need to find information about the place of death of Manoel de Oliveira.\nSTEP 1: Retrieve the relevant documents about `Where was the place of death of Manoel de Oliveira?`.\nSTEP 2: Retrieve the place of death of Manoel de Oliveira from the retrieved documents and provide the answer.""",

    """Question: When was the director of film My Official Wife (1914 Film) born?\nTo address the question, we'll break it down into the following core components:\nIdentify the Film: We need to confirm that "My Official Wife" is indeed the film in question, as there may be multiple films with similar titles.\nIdentify the Director: We need to determine who directed the 1914 film "My Official Wife."\nFind Birth Date: The final goal is to ascertain the birth date of the identified director.\nSTEP 1: Search for detailed information about the film "My Official Wife" (1914). This includes its cast, crew, and particularly the director.\nSTEP 2: From the retrieved information about the film, extract the name of the director.\nSTEP 3: Once the director is identified, we need to search for biographical information on that person to find their birth date.\nSTEP 4: Finalize the response by compiling the birth date and ensuring it is attributed to the correct director of "My Official Wife" (1914).""",

    # hotpotqa
    """Question: Who wrote the obituary for the man who created the \"Watch Mr. Wizard\" television programming?\nQuestion Analysis:
Identify the Creator: The core component of the question is to identify the individual who created the "Watch Mr. Wizard" television program.
Obtain Obituary Details: Once the creator is identified, find and retrieve the obituary that was written for this individual.
Determine the Author of the Obituary: Finally, determine who authored the obituary for the creator of the program.
Step 1: Retrieve information about `who created the \"Watch Mr. Wizard\" television programming?`.
Step 2: Identify the creator of the program from the retrieved documents.
Step 3: Retrieve the information about who wrote the obituary for for the creator.
Step 4: Determine the author of the obituary.""",

    """Question: Where both Games Magazine and The General published by Games Publications?\nKey Information Needed:
Verification if "Games Magazine" was published by "Games Publications."
Verification if "The General" was published by "Games Publications."
Step 1: Retrieve Information about `Is Games Magazine published by Games Publications?`, and identify the publisher of Games Magazine.
Step 2: Retrieve Information about `Is The General published by Games Publications?`, and identify the publisher of The General.
Step 3: Compare the publishers of both magazines to determine if they were published by Games Publications.""",

#     """Question: Where was the father of Mirjam Finkelstein born?\nQuestion Analysis:\n- Core Components:\nSubject: The father of Mirjam Finkelstein\nAction: Determine the birthplace of this individual\n- Additional Information Needed:\nDetails regarding Mirjam Finkelstein's family, particularly her father's background, including his name and place of birth.\nSTEP 1: Conduct a search to gather background information about Mirjam Finkelstein. This may include finding details about her family, specifically focusing on her father.\nSTEP 2: Once sufficient information about Mirjam is gathered, specifically locate the father's name. Conduct a separate search to find detailed information about her father's birthplace.\nSTEP 3: Compile the gathered data into a comprehensive format, ensuring that the birthplace of Mirjam Finkelstein's father is highlighted as the answer to the question.""",

#     """Question: What is the date of death of Joan Of Dampierre's mother?\nQuestion Analysis:
# - Key components of the question:
# We need to find the date of death.
# The person whose death date we're looking for is the mother of Joan of Dampierre.
# - What we need to find out through retrieval:
# The identity of Joan of Dampierre's mother
# The date of death of Joan of Dampierre's mother
# STEP 1: Use the retrieval system to gather information about Joan of Dampierre and look for specific mentions of her mother's name.
# STEP 2: Once the mother's identity is established, use the retrieval system to gather information about her. Focus on finding biographical details, particularly the date of her death.
# STEP 3: Summarize the retrieved information gathered about the date of death of Joan of Dampierre's mother, along with any relevant context or additional details""",

#     """Question: Which film was released first, Welcome To Home Gori or Good Sam?\nQuestion Analysis:
# a. Core components:
# Film 1: "Welcome To Home Gori"
# Film 2: "Good Sam"
# Comparison: Release dates
# b. Additional information needed through retrieval:
# Release date of "Welcome To Home Gori"
# Release date of "Good Sam"
# STEP 1: Use retrieval system to search for "Welcome To Home Gori" as a film title, and find its release date from the retrieved documents.
# STEP 2: Use retrieval system to search for "Good Sam" as a film title, and identify the release date of "Welcome To Home Gori" from the retrieved documents.
# STEP 3: Summarize the release dates of both films and state which film was released first. Provide the answer based on the retrieved information.""",

#     """Question: Where did the director of film Happy Ghost Iv study?\nThe question is asking about the educational background of the director of the film Happy Ghost IV. The primary piece of information needed is the director's education history, specifically where they studied.\nSTEP 1: Retrieve the name of the director who directed Happy Ghost IV.\nSTEP 2: Once the director is identified, focus on retrieving information related to their academic history.\nSTEP 3: Identify the educational institution where the director studied and provide the answer based on the retrieved information.""",

    
]
