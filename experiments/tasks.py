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
