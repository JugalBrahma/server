"""
Evaluation datasets for LangSmith benchmark testing.
"""

examples = [
    {
        "inputs": {"question": "Why was the doctor initially unable to travel to his sick patient?"},
        "outputs": {"expected_answer": "The doctor's horse had died during the night from the severe winter cold, and his servant girl could not borrow a replacement from anyone in the village."}
    },
    {
        "inputs": {"question": "Where did the doctor unexpectedly find the horses and the groom?"},
        "outputs": {"expected_answer": "He found them inside an unused pigsty in his yard after kicking open its rickety door."}
    },
    {
        "inputs": {"question": "What violent act did the groom commit against the servant girl, Rosa?"},
        "outputs": {"expected_answer": "The groom seized Rosa, slammed his face against hers, and bit her cheek, leaving two red rows of tooth marks."}
    },
    {
        "inputs": {"question": "What was the doctor's initial diagnosis of the sick boy upon his first examination?"},
        "outputs": {"expected_answer": "The doctor initially judged the boy to be healthy, noting only sluggish circulation and that his mother had given him too much coffee."}
    },
    {
        "inputs": {"question": "What did the doctor discover upon closer re-examination of the boy's side?"},
        "outputs": {"expected_answer": "He discovered an open wound the size of a palm on the boy's right hip, teeming with writhing rosy worms with white heads and legs."}
    },
    {
        "inputs": {"question": "What did the family and the village elders do to the doctor after his diagnosis?"},
        "outputs": {"expected_answer": "They stripped the doctor naked and laid him into bed beside the patient against the side of the wound while a choir sang outside."}
    },
    {
        "inputs": {"question": "How did the doctor's return journey differ from his initial trip to the patient?"},
        "outputs": {"expected_answer": "The journey out was instantaneous, but the return journey was painfully slow through the freezing snow with the doctor naked and unable to reach his coat."}
    }
]