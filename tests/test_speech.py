from agent.speech import speakable

LEX = {"IELTS": "eye elts", "UOW": "U O W", "UOW India": "U O W India", "GIFT City": "Gift City"}


def test_phone_numbers_are_read_digit_by_digit():
    assert speakable("WhatsApp +91 97734 43748 today") == "WhatsApp plus 9 1, 9 7 7 3 4, 4 3 7 4 8 today"
    assert speakable("call 555-0100") == "call 555-0100"  # too short to be a full number


def test_email_and_website_are_spelled_naturally():
    assert speakable("email UOWI-Admit@uow.edu.au.") == "email U O W I dash admit at U O W dot E D U dot A U."
    assert speakable("see uow.edu.au/india for more") == "see U O W dot E D U dot A U slash india for more"


def test_money_percent_and_pin():
    assert speakable("AUD 27,900 per year") == "27,900 Australian dollars per year"
    assert speakable("₹25 lakh") == "25 lakh rupees"
    assert speakable("Rs. 1,50,000 per year") == "1 lakh 50 thousand rupees per year"
    assert speakable("₹1,25,00,000") == "1 crore 25 lakh rupees"
    assert speakable("a 40% reduction") == "a 40 percent reduction"
    assert speakable("Gujarat, PIN code 382355") == "Gujarat, PIN code 3 8 2 3 5 5"


def test_ordinary_numbers_untouched():
    text = "It has 72 credit points, takes 1.5 years and started in 2026."
    assert speakable(text) == text


def test_lexicon_whole_words_longest_first():
    assert speakable("UOW India needs IELTS 6.5 in GIFT City", LEX) == "U O W India needs eye elts 6.5 in Gift City"
    assert speakable("UOWI and UOWX stay", LEX) == "UOWI and UOWX stay"
