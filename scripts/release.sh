#!/usr/bin/env bash
# release.sh — cut a release whose version is DERIVED from Conventional Commits.
#
# Never pick a version by hand again: git-cliff reads the commit types since the
# last tag and computes the next semver (feat -> minor, fix -> patch, etc., per
# the [bump] policy in cliff.toml). This is what prevents another "features went
# out as a patch" mistake.
#
# Does the LOCAL steps only (changelog + commit + tag). Pushing and publishing
# to PyPI stay manual on purpose — see the printed next-steps.
#
# Usage:  scripts/release.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LAST=$(git describe --tags --abbrev=0 2>/dev/null || echo "(none)")
NEXT=$(uvx git-cliff --bumped-version 2>/dev/null)

if [ "$NEXT" = "$LAST" ]; then
  echo "Nothing to release: no bump-worthy commits since $LAST."
  echo "(Only chore/ci/style or unconventional commits don't move the version.)"
  exit 0
fi

echo "Last tag:     $LAST"
echo "Next version: $NEXT   (computed from commit types since $LAST)"
echo
echo "Commits that will be released:"
git log "${LAST}..HEAD" --oneline 2>/dev/null || git log --oneline
echo

read -rp "Regenerate changelog, commit, and tag $NEXT? [y/N] " ok
[ "$ok" = "y" ] || { echo "Aborted."; exit 1; }

# --bump makes git-cliff label the unreleased section with the computed version.
uvx git-cliff --bump -o CHANGELOG.md
git add CHANGELOG.md

# Keep CITATION.cff in lockstep with the release so the GitHub "Cite this
# repository" widget and any Zenodo archive report the right version/date.
# Tags are vX.Y.Z; CFF wants a bare X.Y.Z, so strip the leading "v".
if [ -f CITATION.cff ]; then
  CFF_VERSION="${NEXT#v}"
  sed -i -E "s/^version: .*/version: ${CFF_VERSION}/" CITATION.cff
  sed -i -E "s/^date-released: .*/date-released: $(date +%F)/" CITATION.cff
  git add CITATION.cff
fi

git commit -m "chore(release): $NEXT"
git tag "$NEXT"

echo
echo "Tagged $NEXT locally. To publish:"
echo "  git push && git push --tags        # push commit + tag"
echo "  # then your usual build/upload, e.g.:"
echo "  # uv build && uv publish            (setuptools-scm reads the tag)"
