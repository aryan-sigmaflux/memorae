# Memory Feature Test Cases

This document lists test cases for the Memory (Knowledge Base) feature, covering saving (remembering), searching (recalling), fuzzy semantic matching, media handling, and deletion (forgetting).

## 1. Simple Saving (Remember)
- "Remember that my passport number is A1234567"
- "Note: my Wi-Fi password is 'supersecret'"
- "Save this: John's birthday is June 15"
- "Write down that I like my coffee black"
- "Remember my blood group is O positive"

## 2. Semantic Retrieval (Recall)
- "What is my passport number?"
- "Do you remember my Wi-Fi password?"
- "When is John's birthday?"
- "How do I like my coffee?"
- "Find the note about my blood group"

## 3. Media Saving (Images/Documents)
- [Upload Image] + "Save this receipt for taxes"
- [Upload PDF] + "Remember this medical report"
- [Upload Image] + "Store this photo of my new car"
- [Upload Voice] + "Note: this is the message from the landlord"

## 4. Media Retrieval
- "Show me the receipt I saved"
- "Find that medical report"
- "Send me the photo of the car"
*Note: System should send the file natively if (LOCAL_PATH: ...) is found.*

## 5. Conditional Saving (Pending Saves)
- [Upload Image] without text -> AI asks: "Want me to save this?"
  - User: "Yes, save it" -> Intent: `confirm_save`
  - User: "No, ignore it" -> Intent: `chat` / clears pending
  - User: "What is in it?" -> AI describes from OCR context but doesn't save yet

## 6. Deletion (Forget)
- "Forget my passport number"
- "Delete the note about my Wi-Fi password"
- "Remove the entry for John's birthday"
- "Clear all notes about travel" (Testing batch-like deletion logic if supported)

## 7. Complex Retrieval (Fuzzy & Multi-Source)
- "Tell me about my documents" (Should list relevant KB entries based on similarity)
- "Do I have any notes about travel?"
- "What did I say about my coffee?"
- "Find everything related to 'John'"

## 8. Intent Precision & Edge Cases
- "I need to save $100" (Should be CHAT, not REMEMBER)
- "Send the file I saved earlier" (Should be RECALL)
- "Search for 'xyz123'" (When no result exists)
- "Remember" (Empty remember intent)

## 9. Semantic Overlap
- User saves: "My car is a blue Tesla"
- User saves: "My wife has a red Tesla"
- User asks: "What color is my car?" (Testing if AI picks the correct specific entry)
