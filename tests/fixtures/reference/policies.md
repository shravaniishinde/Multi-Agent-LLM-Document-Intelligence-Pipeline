# Document processing policies

## Invoice required fields
Every invoice must include an invoice number in the form INV-####, the customer name,
the customer ID in the form CUST-####, the invoice date, the total amount and an ISO
currency code. When subtotal and tax are printed, subtotal plus tax must equal the total.

## Receipt rules
Receipts use RCT-#### reference numbers. A receipt must name the paying customer and
customer ID. The currency may be shown as a symbol next to the total instead of a
separate currency line.

## Approval limits
Each customer has an approval limit in its billing currency. Any invoice or receipt with
a total above that limit requires manager approval and must be escalated to human review,
even when every field is valid.

## Currency policy
Customers are invoiced only in their contracted billing currency. A document in a
different currency must be escalated; it is never corrected automatically.

## OCR quality
Scans with OCR confidence below 0.80 are likely to contain misread characters such as
O for 0 or l for 1. Re-scan with enhanced preprocessing before rejecting such a document.
Documents whose confidence stays low after re-scanning go to human review.
