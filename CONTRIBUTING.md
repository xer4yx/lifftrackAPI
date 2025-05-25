# Contributing to LiftTrack API

## Automatic Version Bumping

This project uses automatic version bumping based on conventional commit messages. When you push commits to the `master` or `main` branch, the version will automatically be bumped based on your commit messages.

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Version Bump Rules

The version bump level is determined by your commit messages:

- **Patch (1.0.0 → 1.0.1)**: 
  - `fix:` - Bug fixes
  - `perf:` - Performance improvements

- **Minor (1.0.0 → 1.1.0)**: 
  - `feat:` - New features

- **Major (1.0.0 → 2.0.0)**: 
  - `BREAKING CHANGE:` in commit body or footer
  - `feat!:` or `fix!:` (with exclamation mark)

- **No version bump**:
  - `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `test:`

### Examples

```bash
# Patch bump
git commit -m "fix: resolve authentication issue"

# Minor bump  
git commit -m "feat: add new workout tracking endpoint"

# Major bump
git commit -m "feat!: redesign API authentication system"

# Or with breaking change footer
git commit -m "feat: add new user management system

BREAKING CHANGE: The user authentication endpoints have been completely redesigned"

# No version bump
git commit -m "docs: update README with new examples"
```

### Workflow Process

When you push to `master`/`main`, the workflow will:

1. **Run Tests**: Ensure all tests pass before bumping version
2. **Analyze Commits**: Check commit messages since last release
3. **Bump Version**: Update version in `version.py` based on commit types
4. **Update Changelog**: Generate changelog from commit messages
5. **Create Tag**: Create a new git tag with the version
6. **Update README**: Run the version update script
7. **Create Release**: Create a GitHub release with changelog

### Manual Testing

Before pushing, you can test the version bump locally:

```bash
# Install semantic-release
pip install python-semantic-release

# Preview what version would be bumped (dry run)
semantic-release version --print

# See what the next version would be
semantic-release version --print-only
```

### Skipping CI

If you need to make commits that shouldn't trigger the workflow, add `[skip ci]` to your commit message:

```bash
git commit -m "docs: minor typo fix [skip ci]"
``` 