one_shot_rag_qa_docs = (
    """Wikipedia Title: Storm of 1897\nThe storm of 1897 was a severe storm that struck the lower to central North Island of New Zealand on 16 April 1897. It caused the ship "Zuleika" to run aground near Cape Palliser, with the loss of 12 lives, and severe flooding. At Clive the flooding caused the loss of a further 12 lives and one person was drowned near Kapiti. There were six further unconfirmed reports of drowning, bringing the total loss of life directly related to the storm event to between 25 and 31. Based on descriptions of the storm, particularly that of Captain Marten of the "Waiapu", it may have been an extratropical cyclone.\n"""
    """Wikipedia Title: MV Bessel\nBessel was a cargo ship that was built in 1925 as Sorrento by AG Weser, Bremen for German owners. She was sold in 1926 and renamed "Bessel". She was seized by the Allies in Vigo, Spain, in May 1945, passed to the Ministry of War Transport (MoWT) and renamed Empire Coniston. In 1946, she was lent to the Danish Government and was allocated to them in 1947. She was sold into Danish merchant service and renamed Birgitte Skou. In 1959, she was sold to Italy and renamed N Martini. She was renamed Nicolo Martini in 1961, serving until 1972 when she ran aground at Portoscuso, Sardinia. Although refloated she was declared a total loss and was scrapped in 1973.\n"""
    """Wikipedia Title: List of shipwrecks in the Bristol Channel\nA number of ships have run aground or sunk in the Bristol Channel, a stretch of water between southern Wales and Somerset. Cardiff, Barry and Penarth were once the largest coal exporters in the world and the channel received significant traffic at the beginning of the twentieth century during exportation.\n"""
    """Wikipedia Title: Sardinia\nSardinia ( ; Italian: "Sardegna" ] , Sardinian: "Sardìgna/Sardìnnia" ] /] , Sassarese: "Sardhigna", Gallurese: "Saldigna", Catalan: "Sardenya", Tabarchino: "Sardegna") is the second largest island in the Mediterranean Sea (after Sicily and before Cyprus) and an autonomous region of Italy. It is located in the Western Mediterranean, to the immediate south of the French island of Corsica.\n"""
    """Wikipedia Title: List of shipwrecks in the Bristol Channel\nA number of ships have run aground or sunk in the Bristol Channel, a stretch of water between southern Wales and Somerset. Cardiff, Barry and Penarth were once the largest coal exporters in the world and the channel received significant traffic at the beginning of the twentieth century during exportation."""
)


rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and verify whether a claim is supported or not. '
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
    'Conclude with "Answer: " to present "SUPPORTED" or "NOT_SUPPORTED", devoid of additional elaborations.'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nVerify the following claim: "
    "The MV Bessel ran aground at the second largest island in the Mediterranean in 1972."
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The MV Bessel, while renamed Nicolo Martini, ran aground at Portoscuso, Sardinia, in 1972. Sardinia is identified as the second largest island in the Mediterranean Sea. "
    "\nAnswer: SUPPORTED"
)

two_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nVerify the following claim: "
    "The MV Bessel ran aground at the second largest island in the Mediterranean in summer of 1972."
    '\nThought: '
)

two_shot_rag_qa_output = (
    "The MV Bessel, later renamed Nicolo Martini, ran aground in Sardinia in 1972. Sardinia is confirmed to be the second largest island in the Mediterranean. The text does not state that the event occurred in the summer."
    "\nAnswer: NOT_SUPPORTED"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": two_shot_rag_qa_input},
    {"role": "assistant", "content": two_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
