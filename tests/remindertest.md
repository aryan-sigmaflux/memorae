# Reminder Feature Test Cases

This document lists test cases for the Reminder feature, covering natural language parsing, relative time resolution, recurrence, and correction logic.

## 1. Simple Relative Time
- "Remind me in 2 minutes"
- "remind me in 5 hours to call John"
- "Remind me in 30 seconds" (Edge case: very short duration)

## 2. Relative Days (Implicit Time)
- "Remind me tomorrow" (Should default to a morning time like 9 AM)
- "set a reminder for Tuesday"
- "remind me next week"
- "in 3 days remind me to check the mail"

## 3. Specific Date & Time
- "Remind me on March 25 at 5pm"
- "remind me tomorrow at 8am"
- "Set a reminder for 2026-12-31 at 23:59"

## 4. Complex Natural Language
- "Hey Memo, can you remind me to pick up the laundry at 6 PM this Friday?"
- "don't forget to remind me to take my medicine at 8 am and 8 pm" (Testing if it handles multiple, though current handles one at a time)

## 5. Recurring Reminders
- "Remind me every day at 9am to drink water"
- "remind me every Monday at 10am to join the standup"
- "Set a reminder for every hour"
- "remind me every night at 11pm to lock the door"

## 6. Correction Logic
- User: "Remind me at 5pm"
- AI: "Reminder set for 5:00 PM."
- User: "no, i meant 6pm" (Should update the last reminder)

## 7. Ambiguous or Invalid Inputs
- "Remind me later" (Testing how LLM/dateparser handles vagueness)
- "remind me on 31st February" (Invalid date)
- "Remind me in -5 minutes" (Past time)
- "Remind me at 25:00" (Invalid time)

## 8. Timezone Specifics
- Testing "5pm" when it's currently 4:30 PM (Should be today)
- Testing "5pm" when it's currently 6:00 PM (Should be tomorrow)
