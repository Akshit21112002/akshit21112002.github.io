# Akshit Singh — personal website

Four static pages, no build step. Every page has its own live, generated background, drawn as points:

| File | Page | Background art (`data-scene` on `<body>`) |
|---|---|---|
| `index.html` | About + Places gallery | `terrain`: a LiDAR-style scan of drifting ridges |
| `publications.html` | Publications | `landscape`: optimizers doing gradient descent on a loss landscape |
| `journey.html` | Experience & mentors | `network`: forward signals and backpropagated errors in a neural net |
| `cv.html` | CV (embedded PDF) | `interference`: two-source wave interference (cursor adds a third source) |

You can swap scenes between pages by changing `data-scene="..."`.
All art lives in `assets/js/main.js`; the captions are the `title` and `text` of each scene there.

## Things to edit

- **Headshot:** save your photo as `assets/img/headshot.jpg`. A portrait crop (4:5, about 800 x 1000 px) works best. Until then, an "AS" monogram is shown.
- **Places gallery (About page):** put your videos in `videos/` as `place-1.mp4` ... `place-6.mp4`, then edit each name and caption in `index.html` (search for `PLACES`). Copy a `<figure class="place-tile">` block to add more, or delete one to remove it. Tiles with no video are hidden automatically on the live site.
- **Links:** search every `.html` for `href="#"` (paper, code, project page, Google Scholar, GitHub) and paste your URLs.
- **CV:** replace `assets/Akshit_Singh_CV.pdf` with a new version whenever you update it (keep the same filename).
- **Mentor photos (optional):** put images in `assets/img/` and replace the initials, e.g. `<div class="mentor-mark"><img src="assets/img/mirza.jpg" alt=""></div>`.

## Make the videos small (recommended)

The gallery tiles are about 380px wide, so 720px-wide video is plenty and keeps the page fast.
GitHub's browser upload limit is 25 MB per file. With [ffmpeg](https://ffmpeg.org/download.html):

```bash
ffmpeg -i original.mp4 -vf "scale=720:-2,fps=24" -c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p -an -movflags +faststart place-1.mp4
```

## Preview on your computer

```bash
cd path/to/this/folder
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Locally, missing photos and videos show an "Add ..." hint so you know what's left.

## Publish on GitHub Pages

1. Create a **new public repository** named exactly `YOUR-USERNAME.github.io`.
2. Upload everything in this folder (keep the folder structure; include the hidden `.nojekyll` file):
   - **Browser:** *Add file → Upload files*, drag in the contents of this folder, then *Commit changes*.
   - **Git:**
     ```bash
     cd path/to/this/folder
     git init
     git add .
     git commit -m "Personal website"
     git branch -M main
     git remote add origin https://github.com/YOUR-USERNAME/YOUR-USERNAME.github.io.git
     git push -u origin main
     ```
3. In the repo go to *Settings → Pages*, choose *Deploy from a branch*, branch `main`, folder `/ (root)`, and save.
4. After 1–2 minutes, open `https://YOUR-USERNAME.github.io`.
5. To update later, edit or replace files and commit again; the site redeploys automatically.
