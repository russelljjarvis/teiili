# Contributing to teili

## We develop with Gitlab
We use a [hosted Gitlab instance](https://code.ini.uzh.ch) to host code, and to track issues and feature requests.
If you don't have access to https://code.ini.uzh.ch/ncs/teili and you didn't get your copy of teili from there, we'll also consider taking patches.

If you do have access to https://code.ini.uzh.ch/ncs/teili and want to contribute, please
1. Starting from the `dev` branch, create a new topic branch named `dev-`_mytopic_ for your work.
2. Clone the repository to where you're going to work.
3. Copy `.pre-push-hook.sh` to `.git/hooks/pre-push` and make sure it is executable. This will run the test code every time you push, and prevent the push from completing if a test fails!
4. Code away to your heart's content. Note that we use the [PEP 8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/). Don't forget to write tests too.
5. Run `pycodestyle` over your code to check that it conforms to PEP 8.
6. Update the documentation if necessary.
7. Commit often, using good commit messages. (See https://chris.beams.io/posts/git-commit/)
8. Push back to code.ini.uzh.ch fairly often too, to enable the rest of us to see what you're up to.
9. Make sure that the CI pipeline runs successfully.
10. When you're done, make a merge request.

## Any contributions you make will be under the MIT Software License
In contributing back to https://code.ini.uzh.ch/ncs/teili, you agree to license your contribution under the MIT license, see the `LICENSE` file.
In particular, if you are employed by an organisation which claims rights over IP created by you as a consequence of that employment, it is _*your*_ responsibilty to check that you are allowed to license your contribution under the MIT license _*before*_ you push it to code.ini.uzh.ch.

## Report bugs using Gitlab's issues
We use our Gitlab [Issues](https://code.ini.uzh.ch/ncs/teili/issues) to track bugs and enhancement requests.

Report a bug or make an enhancement request by opening a [new issue](https://code.ini.uzh.ch/ncs/teili/issues/new?issue); it's that easy!

Bug reports should be written with plenty of detail, background and sample code.
Good bug reports tend to have:
- A quick summary and/or background;
- Specific steps to reproduce;
- Sample code if possible;
- A description of what you expected should happen;
- A description of what actually happens;
- A list of things you tried that didn't work.

## Acknowledgments
This document was based on [Brian A. Danielak's template for contribution guidelines](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62) which in turn was adapted from the [Draft.js CONTRIBUTING.md](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).

