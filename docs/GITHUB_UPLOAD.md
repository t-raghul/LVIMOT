# Publish LVIMOT on GitHub

1. Create a new **Public** repository named `LVIMOT` at
   https://github.com/new . Do **not** add a GitHub-generated README,
   `.gitignore`, or license.
2. Extract the provided GitHub-ready ZIP. Open a terminal in the **LVIMOT**
   folder containing `README.md`.
3. Review staged files and push (replace `YOUR_USERNAME`):

```bash
git init
git add .
git status
git commit -m "Initial public LVIMOT release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/LVIMOT.git
git push -u origin main
```

If Git asks for your name/email, configure them first. GitHub may ask you
to sign in through your browser or Git Credential Manager.

4. On the repository page, select **Releases** > **Draft a new release**.
   Create tag `v1.0.0`, title `LVIMOT v1.0.0`, attach the same ZIP under
   release assets, and publish. GitHub also creates its own source ZIP.

Before publishing, verify no confidential material is included and that you
have the rights to distribute the included model artifacts. The third-party
model licenses are not replaced by any GitHub release tag.
