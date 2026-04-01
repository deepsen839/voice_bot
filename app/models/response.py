import random

RESPONSES = {
    "order_status": [
        "Your order is currently being processed.",
        "Your order is on the way.",
    ],
    "cancel_order": [
        "Your order has been successfully cancelled.",
        "We've cancelled your order as requested.",
    ],
    "refund": [
        "Your refund has been initiated.",
        "Refund process has started.",
    ],
    "subscription": [
        "Your subscription details have been updated.",
    ],
    "technical_issue": [
        "We are working on resolving the issue.",
    ],
    "payment_issue": [
        "There seems to be a payment issue. Please check your details.",
    ],
    "greeting": [
        "Hello! How can I assist you today?",
    ],
    "goodbye": [
        "Goodbye! Have a great day!",
    ],
    "complaint": [
        "We're sorry for the inconvenience caused.",
    ],
    "other": [
        "Could you please provide more details?"
    ]
}

def generate_response(intent):
    intent = intent.strip().lower()
    return random.choice(RESPONSES.get(intent, ["Sorry, I didn't understand."]))