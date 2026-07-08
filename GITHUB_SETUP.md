# Pushing this repo to GitHub

## 1. Create the repository on GitHub
Go to https://github.com/new, name it (e.g. `transformation-invariant-gradcam`),
choose Public or Private, and **do not** add a README/.gitignore/license there
(this folder already has them). Click *Create repository*.

## 2. Initialize and push from this folder
From inside the `transformation-invariant-gradcam/` folder:

```bash
git init
git add .
git commit -m "Initial commit: thesis, presentation, and reconstructed code scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/transformation-invariant-gradcam.git
git push -u origin main
```

Replace `<your-username>` with your GitHub handle.

## Notes

- **Large PDF.** `docs/Thesis_Paper.pdf` is ~31 MB. That is under GitHub's 100 MB
  hard limit and pushes fine over HTTPS. If you would rather keep the git history
  light, track PDFs with [Git LFS](https://git-lfs.com):
  ```bash
  git lfs install
  git lfs track "*.pdf"
  git add .gitattributes
  ```
- **Datasets and results are ignored.** `.gitignore` keeps `data/` contents, model
  weights, and generated `results/` out of the repo. The empty folders are preserved
  via `.gitkeep` files.
- **Update the placeholders.** Set your name/handle/year in `LICENSE`, `CITATION.cff`,
  and the footer of `README.md`, and fill the `TODO` markers in `src/` and
  `experiments/` as you rebuild the lost code.
