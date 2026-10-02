"""Task definitions shared by experiments: the question and labels, with and without
descriptions. Every benchmark arm uses these exact strings."""

SUPPORT_QUESTION = "Which team should handle this customer support message?"

# Bitext's 11 customer-support categories. Descriptions follow Bitext's semantics: "cancel" is
# cancellation FEES only (cancelling an order is "order"); "subscription" is newsletters/promos.
SUPPORT_LABELS_DESCRIBED = {
    "account": "create, delete or edit an account; password or sign-up problems",
    "order": "placing, cancelling, changing or tracking an order",
    "refund": "refund policy, getting a refund, or refund status",
    "invoice": "getting or checking an invoice or bill",
    "contact": "reaching customer service or a human agent",
    "payment": "payment methods or problems paying",
    "feedback": "complaints or reviews",
    "delivery": "delivery options and how long delivery takes",
    "shipping": "setting up or changing a shipping address",
    "subscription": "newsletter or promotional email/SMS subscribe or unsubscribe",
    "cancel": "questions about cancellation FEES or charges only",
}
SUPPORT_LABELS = list(SUPPORT_LABELS_DESCRIBED)

# Benchmark tasks built by prepare_bench.py. Each has a question and labels with short
# descriptions; the "names" condition uses the label names only.
TASKS = {
    "support": {
        "question": SUPPORT_QUESTION,
        "labels": SUPPORT_LABELS_DESCRIBED,
    },
    # Google Hinglish-TOP: the same voice-assistant request in English and in Hinglish.
    "top": {
        "question": "Which domain does this voice-assistant request belong to?",
        "labels": {
            "alarm": "creating, changing, checking or deleting alarms",
            "event": "finding events or things to do, such as concerts, shows or festivals",
            "messaging": "sending, reading or replying to text messages",
            "music": "playing, pausing, skipping or managing music",
            "navigation": "directions, routes, distance, travel time and traffic",
            "reminder": "creating, changing, checking or deleting reminders",
            "timer": "countdown timers",
            "weather": "weather conditions and forecasts",
        },
    },
    # Amazon MASSIVE scenarios, parallel across English, Hindi, Bengali and Tamil.
    "massive": {
        "question": "Which scenario does this voice-assistant request belong to?",
        "labels": {
            "alarm": "setting, checking or removing alarms",
            "audio": "device volume: louder, quieter or mute",
            "calendar": "setting, checking or removing calendar events and reminders",
            "cooking": "recipes and cooking questions",
            "datetime": "the current date or time, or converting time zones",
            "email": "reading or sending email, or managing email contacts",
            "general": "small talk: greetings, jokes and odd remarks",
            "iot": "smart-home devices: lights, plugs, robot vacuum, coffee machine",
            "lists": "creating, checking or changing to-do and shopping lists",
            "music": "music preferences and information about songs (not playing them)",
            "news": "news headlines and updates",
            "play": "playing music, radio, podcasts, audiobooks or games",
            "qa": "factual questions: definitions, facts, maths, currency, stock prices",
            "recommendation": "recommendations for events, films or places",
            "social": "posting to or checking social media",
            "takeaway": "ordering takeaway food or checking a takeaway order",
            "transport": "trains, taxis, tickets and traffic",
            "weather": "weather conditions and forecasts",
        },
    },
    # AI4Bharat IndicSentiment: product reviews, parallel across English and Indic languages.
    "reviews": {
        "question": "What is the sentiment of this product review?",
        "labels": {
            "positive": "the reviewer is satisfied or praises the product",
            "negative": "the reviewer is dissatisfied or criticises the product",
        },
    },
    # SemEval-2020 Task 9 SentiMix: real code-mixed Hinglish tweets.
    "tweets": {
        "question": "What is the sentiment of this tweet?",
        "labels": {
            "positive": "approving, happy, praising or supportive",
            "negative": "critical, angry, sad or mocking",
            "neutral": "neither clearly positive nor clearly negative, or purely factual",
        },
    },
}


def choices(task: str, condition: str):
    """Choices to pass to Decider.decide: label names only, or label → description."""
    labels = TASKS[task]["labels"]
    return list(labels) if condition == "names" else dict(labels)
