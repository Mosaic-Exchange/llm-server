# llm-server

## Contributing

Please follow the following procedure when making any changes to the repository:

1. [Create an issue on Github.][0]
2. Under "**Development**" on the right side of the issue overview, click "**Create a branch**".
3. In most cases, the default branch name should suffice. If not, **make sure to keep
   the issue number and dash at the beginning of the branch name**.
4. Switch to the new branch locally with
```bash
git fetch origin
git switch [branch name here]
```
   (or by refreshing the repository and selecting the new branch in your IDE).

5. Do the work. Make sure to create automated tests along the way.
6. Push your work and open a pull request (the button should appear on the [repository
   page][1]).
7. Check the results of the CI pipeline in the pull request overview. If there are
   errors, correct them and push the changes.
8. Respond to reviewer feedback.
9. Merge.

[0]: https://github.com/Mosaic-Exchange/llm-server/issues/new/choose
[1]: https://github.com/Mosaic-Exchange/llm-server
