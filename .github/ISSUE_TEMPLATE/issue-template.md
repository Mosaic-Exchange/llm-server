---
name: Issue template
about: Primary issue template used by the project
title: ''
labels: ''
assignees: ''

---

## Description/User story

This is a brief description of the added functionality, e.g. "Currently, the chat menu displays messages from the current session and stores chat messages in memory within the ChatSession object. We want users to be able to view their chat history, which requires that chat history be stored in the database."

## Acceptance criteria

When someone is reviewing the pull request associated with this issue, they will look at this list and make sure everything statement is true. This can also act as a task list for the developer assigned to the issue, so it helps if you write the acceptance criteria in the order in which the functionality will be added. e.g.

AC1: When a user types a message in the chat window and closes the application normally, the chat history is written to a new table in the database (ChatHistory). (QA: Check this in the database viewer.)

AC2: If the user doesn't type anything, nothing is stored in the database. (QA: Check this in the database viewer.)

AC3: New methods have automated (JUnit) tests, and all tests pass.
