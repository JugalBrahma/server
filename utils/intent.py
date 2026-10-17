def is_mcq_request(question: str) -> bool:
    """Detect if user is requesting MCQ generation"""
    triggers = [
        "generate mcq",
        "create mcq", 
        "make quiz",
        "quiz me",
        "test me",
        "multiple choice",
        "mcq",
        "questions",
        "practice questions"
    ]
    
    q = question.lower().strip()
    return any(trigger in q for trigger in triggers)

def is_mcq_answer(answer: str) -> bool:
    """Detect if user is answering an MCQ"""
    answer_indicators = [
        "answer is",
        "the answer is",
        "correct answer",
        "option",
        "choice",
        "a)", "b)", "c)", "d)"
    ]
    
    a = answer.lower().strip()
    return any(indicator in a for indicator in answer_indicators)
