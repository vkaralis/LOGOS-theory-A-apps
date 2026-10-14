# Release v0.5.0

`pyproject.toml`, `logos_apps.__version__`, CHANGELOG and this release use 0.5.0. Upload the contents of this directory to the intended GitHub repository. Do not upload the private review report or the original source archive and documents. Set the About description from `github-about.txt` and the topics from `github-topics.txt` in GitHub settings after upload. These settings are configured separately from repository files.

For a Git checkout, commit the package and create the annotated release tag after verification:

```bash
git add .
git commit -m "Release LOGOS A applications v0.5.0"
git tag -a v0.5.0 -m "LOGOS A applications v0.5.0"
git push origin HEAD
git push origin v0.5.0
```

If `v0.5.0` already exists locally, verify that `git rev-parse HEAD` and `git rev-parse 'v0.5.0^{}'` agree before publishing it. If an unpublished local tag points to an earlier preparation commit, refresh it with `git tag -f -a v0.5.0 -m "LOGOS A applications v0.5.0"`. Published tags should not be replaced; later changes require a new release version. A ZIP upload does not create a Git tag. The tag version test checks the version name; the commit comparison checks the release contents.

Suggested GitHub release title: **LOGOS A applications v0.5.0**. Release notes: reproducible specified-model and synthetic event-analysis workflows; Python 3.12; MIT code and synthetic data; full-training bootstrap and independent numerical checks. Attach the ZIP and its SHA-256 file if desired.
